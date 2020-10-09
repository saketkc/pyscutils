import os
import warnings

warnings.simplefilter("ignore")
import shutil
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import proplot
import scanpy as sc
import scvi
import seaborn as sns
import torch
import torch.nn as nn
from adjustText import adjust_text
from scvi import set_seed
from scvi.dataset import AnnDatasetFromAnnData
from scvi.models.utils import one_hot
from scvi.inference import UnsupervisedTrainer, load_posterior
from scvi.models.distributions import (
    NegativeBinomial,
    Poisson,
    ZeroInflatedNegativeBinomial,
)
from scvi.models.log_likelihood import log_nb_positive, log_zinb_positive
from scvi.models.modules import DecoderSCVI, Encoder, FCLayers, LinearDecoderSCVI
from scvi.models.vae import LDVAE, VAE
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl


## Modifications from scVI code marked with '################ ===>'


def compute_scvi_latent(
    adata: sc.AnnData,
    n_latent: int = 50,
    n_encoder: int = 1,
    n_epochs: int = 200,
    lr: float = 1e-3,
    use_batches: bool = False,
    use_cuda: bool = False,
    linear: bool = False,
    cell_offset: str = "none",
    gene_offset: str = "none",
    ldvae_bias: bool = False,
    reconstruction_loss: str = "zinb",
    hvg_genes=None,
) -> Tuple[scvi.inference.Posterior, np.ndarray]:
    """Train and return a scVI model and sample a latent space

    :param adata: sc.AnnData object non-normalized
    :param n_latent: dimension of the latent space
    :param n_epochs: number of training epochs
    :param lr: learning rate
    :param use_batches
    :param use_cuda
    :return: (scvi.Posterior, latent_space)
    """
    # Convert easily to scvi dataset
    scviDataset = AnnDatasetFromAnnData(adata)
    if isinstance(hvg_genes, int):
        scviDataset.subsample_genes(hvg_genes)
    # print(scviDataset.X.shape)
    # print(scviDataset.X[:10,:5])
    # print(scviDataset.raw.X.shape)
    if isinstance(scviDataset.X, np.ndarray):
        X = scviDataset.X
    else:
        X = scviDataset.X.toarray()
    gene_mean = torch.mean(
        torch.from_numpy(X).float().to(torch.cuda.current_device()), dim=1
    )
    cell_mean = torch.mean(
        torch.from_numpy(X).float().to(torch.cuda.current_device()), dim=0
    )
    # Train a model
    if not linear:
        vae = VAEGeneCell(
            scviDataset.nb_genes,
            n_batch=scviDataset.n_batches * use_batches,
            n_latent=n_latent,
            n_layers=n_encoder,
            cell_offset=cell_offset,
            gene_offset=gene_offset,
            reconstruction_loss=reconstruction_loss,
        )
    else:
        vae = LDVAEGeneCell(
            scviDataset.nb_genes,
            n_batch=scviDataset.n_batches * use_batches,
            n_latent=n_latent,
            n_layers_encoder=n_encoder,
            cell_offset=cell_offset,
            gene_offset=gene_offset,
            bias=ldvae_bias,
            reconstruction_loss=reconstruction_loss,
        )
    trainer = UnsupervisedTrainer(vae, scviDataset, train_size=1.0, use_cuda=use_cuda)
    trainer.train(n_epochs=n_epochs, lr=lr)

    # Extract latent space
    posterior = trainer.create_posterior(
        trainer.model, scviDataset, indices=np.arange(len(scviDataset))
    ).sequential()

    latent, _, _ = posterior.get_latent()

    return posterior, latent, vae, trainer


# Decoder
class DecoderSCVI(nn.Module):
    """Decodes data from latent space of ``n_input`` dimensions ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers

    Returns
    -------
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
        )

        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1)
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(
        self, dispersion: str, z: torch.Tensor, library: torch.Tensor, *cat_list: int
    ):
        """The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        Parameters
        ----------
        dispersion
            One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell
        z :
            tensor with shape ``(n_input,)``
        library
            library size
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        4-tuple of :py:class:`torch.Tensor`
            parameters for the ZINB distribution of expression
        """
        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, *cat_list)
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = (torch.exp(library)) * px_scale  # torch.clamp( , max=12)
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout


## Modifications from scVI code marked with '################ ===>'
class DecoderSCVIGeneCell(DecoderSCVI):
    """Decodes data from latent space of ``n_input`` dimensions ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers

    Returns
    -------
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
    ):
        super().__init__(n_input, n_output, n_cat_list, n_layers, n_hidden)

    def forward(
        self,
        dispersion: str,
        z: torch.Tensor,
        library: torch.Tensor,
        *cat_list: int,
        cell_offset: torch.Tensor,
        gene_offset: torch.Tensor,
        dispersion_clamp: list,
    ):
        """The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        Parameters
        ----------
        dispersion
            One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell
        z :
            tensor with shape ``(n_input,)``
        library
            library size
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        4-tuple of :py:class:`torch.Tensor`
            parameters for the ZINB distribution of expression
        """
        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, *cat_list)
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability
        ################ ===>
        cell_offset = torch.reshape(cell_offset, (cell_offset.shape[0], 1))

        px_rate = (
            (torch.exp(library) * (cell_offset)) * px_scale * gene_offset
        )  # torch.clamp( , max=12)
        px_rate = (
            (torch.exp(library) * (cell_offset)) * px_scale * gene_offset
        )  # torch.clamp( , max=12)
        # px_rate = cell_offset #torch.exp(library) + cell_mean  * px_scale  # torch.clamp( , max=12)
        # px_rate = torch.exp(library + cell_mean) * px_scale  # torch.clamp( , max=12)
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        if dispersion == "gene-cell" and dispersion_clamp:
            px_r = torch.clamp(px_r, min=dispersion_clamp[0], max=dispersion_clamp[1])
        return px_scale, px_r, px_rate, px_dropout


class LinearDecoderSCVIGeneCell(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        use_batch_norm: bool = True,
        bias: bool = False,
    ):
        super(LinearDecoderSCVIGeneCell, self).__init__()

        # mean gamma
        self.factor_regressor = FCLayers(
            n_in=n_input,
            n_out=n_output,
            n_cat_list=n_cat_list,
            n_layers=1,
            use_relu=False,
            use_batch_norm=use_batch_norm,
            bias=bias,
            dropout_rate=0,
        )

        # dropout
        self.px_dropout_decoder = FCLayers(
            n_in=n_input,
            n_out=n_output,
            n_cat_list=n_cat_list,
            n_layers=1,
            use_relu=False,
            use_batch_norm=use_batch_norm,
            bias=bias,
            dropout_rate=0,
        )

    def forward(
        self,
        dispersion: str,
        z: torch.Tensor,
        library: torch.Tensor,
        *cat_list: int,
        cell_offset: torch.Tensor,
        gene_offset: torch.Tensor,
    ):
        # The decoder returns values for the parameters of the ZINB distribution
        raw_px_scale = self.factor_regressor(z, *cat_list)
        px_scale = torch.softmax(raw_px_scale, dim=-1)
        px_dropout = self.px_dropout_decoder(z, *cat_list)

        ##px_rate = torch.exp(library) * px_scale
        ################ ===>
        cell_offset = torch.reshape(cell_offset, (cell_offset.shape[0], 1))
        px_rate = (
            (torch.exp(library) * cell_offset) * px_scale * gene_offset
        )  # torch.clamp( , max=12)
        px_r = None

        return px_scale, px_r, px_rate, px_dropout


# VAEGeneCell model
class VAEGeneCell(nn.Module):
    """Variational auto-encoder model.

    This is an implementation of the scVI model descibed in [Lopez18]_

    Parameters
    ----------
    n_input
        Number of input genes
    n_batch
        Number of batches, if 0, no batch correction is performed.
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    dropout_rate
        Dropout rate for neural networks
    dispersion
        One of the following

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    reconstruction_loss
        One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution

    Examples
    --------

    >>> gene_dataset = CortexDataset()
    >>> vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
    ... n_labels=gene_dataset.n_labels)
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        log_variational: bool = True,
        reconstruction_loss: str = "zinb",
        latent_distribution: str = "normal",
        cell_offset: str = "none",  ################ ===>
        gene_offset: str = "none",  ################ ===>
        dispersion_clamp: list = [],
        beta_disentanglement: float = 1.0,
        kl_type: str = "reverse",
    ):
        super().__init__()
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.reconstruction_loss = reconstruction_loss
        # Automatically deactivate if useless
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.latent_distribution = latent_distribution

        ################ ===>
        self.cell_offset = cell_offset
        self.gene_offset = gene_offset
        self.dispersion_clamp = dispersion_clamp
        self.beta_disentanglement = beta_disentanglement
        self.kl_type = kl_type

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError(
                "dispersion must be one of ['gene', 'gene-batch',"
                " 'gene-label', 'gene-cell'], but input was "
                "{}.format(self.dispersion)"
            )

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
        )
        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = Encoder(
            n_input, 1, n_layers=1, n_hidden=n_hidden, dropout_rate=dropout_rate
        )
        # decoder goes from n_latent-dimensional space to n_input-d data
        ################ ===>
        self.decoder = DecoderSCVIGeneCell(
            n_latent,
            n_input,
            n_cat_list=[n_batch],
            n_layers=n_layers,
            n_hidden=n_hidden,
        )

    def get_latents(self, x, y=None) -> torch.Tensor:
        """Returns the result of ``sample_from_posterior_z`` inside a list

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
        y
            tensor of cell-types labels with shape ``(batch_size, n_labels)`` (Default value = None)

        Returns
        -------
        type
            one element list of tensor

        """
        return [self.sample_from_posterior_z(x, y)]

    def sample_from_posterior_z(
        self, x, y=None, give_mean=False, n_samples=5000
    ) -> torch.Tensor:
        """Samples the tensor of latent values from the posterior

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
        y
            tensor of cell-types labels with shape ``(batch_size, n_labels)`` (Default value = None)
        give_mean
            is True when we want the mean of the posterior  distribution rather than sampling (Default value = False)
        n_samples
            how many MC samples to average over for transformed mean (Default value = 5000)

        Returns
        -------
        type
            tensor of shape ``(batch_size, n_latent)``

        """
        if self.log_variational:
            x = torch.log(1 + x)
        qz_m, qz_v, z = self.z_encoder(x, y)  # y only used in VAEC
        if give_mean:
            if self.latent_distribution == "ln":
                samples = Normal(qz_m, qz_v.sqrt()).sample([n_samples])
                z = self.z_encoder.z_transformation(samples)
                z = z.mean(dim=0)
            else:
                z = qz_m
        return z

    def sample_from_posterior_l(self, x) -> torch.Tensor:
        """Samples the tensor of library sizes from the posterior

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
        y
            tensor of cell-types labels with shape ``(batch_size, n_labels)``

        Returns
        -------
        type
            tensor of shape ``(batch_size, 1)``

        """
        if self.log_variational:
            x = torch.log(1 + x)
        ql_m, ql_v, library = self.l_encoder(x)
        return library

    def get_sample_scale(
        self, x, batch_index=None, y=None, n_samples=1, transform_batch=None
    ) -> torch.Tensor:
        """Returns the tensor of predicted frequencies of expression

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size`` (Default value = None)
        y
            tensor of cell-types labels with shape ``(batch_size, n_labels)`` (Default value = None)
        n_samples
            number of samples (Default value = 1)
        transform_batch
            int of batch to transform samples into (Default value = None)

        Returns
        -------
        type
            tensor of predicted frequencies of expression with shape ``(batch_size, n_input)``

        """
        return self.inference(
            x,
            batch_index=batch_index,
            y=y,
            n_samples=n_samples,
            transform_batch=transform_batch,
        )["px_scale"]

    def get_sample_rate(
        self, x, batch_index=None, y=None, n_samples=1, transform_batch=None
    ) -> torch.Tensor:
        """Returns the tensor of means of the negative binomial distribution

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
        y
            tensor of cell-types labels with shape ``(batch_size, n_labels)`` (Default value = None)
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size`` (Default value = None)
        n_samples
            number of samples (Default value = 1)
        transform_batch
            int of batch to transform samples into (Default value = None)

        Returns
        -------
        type
            tensor of means of the negative binomial distribution with shape ``(batch_size, n_input)``

        """
        return self.inference(
            x,
            batch_index=batch_index,
            y=y,
            n_samples=n_samples,
            transform_batch=transform_batch,
        )["px_rate"]

    def get_reconstruction_loss(
        self, x, px_rate, px_r, px_dropout, **kwargs
    ) -> torch.Tensor:
        # Reconstruction Loss
        px_rate_ = px_rate

        if self.reconstruction_loss == "zinb":
            reconst_loss = (
                -ZeroInflatedNegativeBinomial(
                    mu=px_rate_, theta=px_r, zi_logits=px_dropout
                )
                .log_prob(x)
                .sum(dim=-1)
            )
        elif self.reconstruction_loss == "nb":
            reconst_loss = (
                -NegativeBinomial(mu=px_rate_, theta=px_r).log_prob(x).sum(dim=-1)
            )
        elif self.reconstruction_loss == "poisson":
            reconst_loss = -Poisson(px_rate_).log_prob(x).sum(dim=-1)
        return reconst_loss

    def inference(
        self, x, batch_index=None, y=None, n_samples=1, transform_batch=None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Helper function used in forward pass"""
        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)

        # Sampling
        qz_m, qz_v, z = self.z_encoder(x_, y)
        ql_m, ql_v, library = self.l_encoder(x_)

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            # when z is normal, untran_z == z
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.z_encoder.z_transformation(untran_z)
            ql_m = ql_m.unsqueeze(0).expand((n_samples, ql_m.size(0), ql_m.size(1)))
            ql_v = ql_v.unsqueeze(0).expand((n_samples, ql_v.size(0), ql_v.size(1)))
            library = Normal(ql_m, ql_v.sqrt()).sample()

        if transform_batch is not None:
            dec_batch_index = transform_batch * torch.ones_like(batch_index)
        else:
            dec_batch_index = batch_index

        ################ ===>
        try:  # if use_cuda:
            cell_offset = torch.ones(x.shape[0]).to(torch.cuda.current_device())
            gene_offset = torch.ones(x.shape[1]).to(torch.cuda.current_device())
        except:
            cell_offset = torch.ones(x.shape[0])
            gene_offset = torch.ones(x.shape[1])

        if self.cell_offset == "count":
            cell_offset = torch.sum(x, dim=1)
        elif self.cell_offset == "mean":
            cell_offset = torch.mean(x, dim=1)
        if self.gene_offset == "count":
            gene_offset = torch.sum(x, dim=0)
        elif self.gene_offset == "mean":
            gene_offset = torch.mean(x, dim=0)

        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion,
            z,
            library,
            dec_batch_index,
            y,
            cell_offset=cell_offset,  ################ ===>
            gene_offset=gene_offset,  ################ ===>
            dispersion_clamp=self.dispersion_clamp,
        )
        if self.dispersion == "gene-label":
            px_r = F.linear(
                one_hot(y, self.n_labels), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(dec_batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r
        px_r = torch.exp(px_r)

        return dict(
            px_scale=px_scale,
            px_r=px_r,
            px_rate=px_rate,
            px_dropout=px_dropout,
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
            ql_m=ql_m,
            ql_v=ql_v,
            library=library,
        )

    def forward(
        self, x, local_l_mean, local_l_var, batch_index=None, y=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the reconstruction loss and the KL divergences

        Parameters
        ----------
        x
            tensor of values with shape (batch_size, n_input)
        local_l_mean
            tensor of means of the prior distribution of latent variable l
            with shape (batch_size, 1)
        local_l_var
            tensor of variancess of the prior distribution of latent variable l
            with shape (batch_size, 1)
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size`` (Default value = None)
        y
            tensor of cell-types labels with shape (batch_size, n_labels) (Default value = None)

        Returns
        -------
        type
            the reconstruction loss and the Kullback divergences

        """
        # Parameters for z latent distribution
        outputs = self.inference(x, batch_index, y)
        qz_m = outputs["qz_m"]
        qz_v = outputs["qz_v"]
        ql_m = outputs["ql_m"]
        ql_v = outputs["ql_v"]
        px_rate = outputs["px_rate"]
        px_r = outputs["px_r"]
        px_dropout = outputs["px_dropout"]

        # KL Divergence
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)

        # only use it on mean
        if self.kl_type == "reverse":
            kl_divergence_z = kl(
                Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)
            ).sum(dim=1)
        elif self.kl_type == "forward":
            kl_divergence_z = kl(
                Normal(mean, scale), Normal(qz_m, torch.sqrt(qz_v))
            ).sum(dim=1)
        elif self.kl_type == "symmetric":
            p_sum_q = Normal(mean + qz_m, scale + torch.sqrt(qz_v))
            kl_divergence_z_f = kl(Normal(mean, scale), p_sum_q).sum(dim=1)
            kl_divergence_z_r = kl(Normal(qz_m, torch.sqrt(qz_v)), p_sum_q).sum(dim=1)
            kl_divergence_z = 0.5 * (kl_divergence_z_f + kl_divergence_z_r)

        kl_divergence_l = kl(
            Normal(ql_m, torch.sqrt(ql_v)),
            Normal(local_l_mean, torch.sqrt(local_l_var)),
        ).sum(dim=1)

        kl_divergence = kl_divergence_z * self.beta_disentanglement

        reconst_loss = self.get_reconstruction_loss(
            x,
            px_rate,
            px_r,
            px_dropout,
        )

        return reconst_loss + kl_divergence_l, kl_divergence, 0.0


class LDVAEGeneCell(VAEGeneCell):
    """Linear-decoded Variational auto-encoder model.

    Implementation of [Svensson20]_.

    This model uses a linear decoder, directly mapping the latent representation
    to gene expression levels. It still uses a deep neural network to encode
    the latent representation.

    Compared to standard VAE, this model is less powerful, but can be used to
    inspect which genes contribute to variation in the dataset. It may also be used
    for all scVI tasks, like differential expression, batch correction, imputation, etc.
    However, batch correction may be less powerful as it assumes a linear model.

    Parameters
    ----------
    n_input
        Number of input genes
    n_batch
        Number of batches
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer (for encoder)
    n_latent
        Dimensionality of the latent space
    n_layers_encoder
        Number of hidden layers used for encoder NNs
    dropout_rate
        Dropout rate for neural networks
    dispersion
        One of the following

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    reconstruction_loss
        One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
    use_batch_norm
        Bool whether to use batch norm in decoder
    bias
        Bool whether to have bias term in linear decoder
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers_encoder: int = 1,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        log_variational: bool = True,
        reconstruction_loss: str = "nb",
        use_batch_norm: bool = True,
        bias: bool = False,
        latent_distribution: str = "normal",
        cell_offset: str = "none",
        gene_offset: str = "none",
    ):
        super().__init__(
            n_input,
            n_batch,
            n_labels,
            n_hidden,
            n_latent,
            n_layers_encoder,
            dropout_rate,
            dispersion,
            log_variational,
            reconstruction_loss,
            latent_distribution,
            cell_offset,  ################ ===>
            gene_offset,  ################ ===>
        )
        self.use_batch_norm = use_batch_norm
        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_layers=n_layers_encoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
        )
        ################ ===>
        self.decoder = LinearDecoderSCVIGeneCell(
            n_latent,
            n_input,
            n_cat_list=[n_batch],
            use_batch_norm=use_batch_norm,
            bias=bias,
        )

    @torch.no_grad()
    def get_loadings(self) -> np.ndarray:
        """Extract per-gene weights (for each Z, shape is genes by dim(Z)) in the linear decoder."""
        # This is BW, where B is diag(b) batch norm, W is weight matrix
        if self.use_batch_norm is True:
            w = self.decoder.factor_regressor.fc_layers[0][0].weight
            bn = self.decoder.factor_regressor.fc_layers[0][1]
            sigma = torch.sqrt(bn.running_var + bn.eps)
            gamma = bn.weight
            b = gamma / sigma
            bI = torch.diag(b)
            loadings = torch.matmul(bI, w)
        else:
            loadings = self.decoder.factor_regressor.fc_layers[0][0].weight
        loadings = loadings.detach().cpu().numpy()
        if self.n_batch > 1:
            loadings = loadings[:, : -self.n_batch]

        return loadings


def compute_scvi_latent(
    adata: sc.AnnData,
    n_latent: int = 50,
    n_encoder: int = 1,
    n_epochs: int = 200,
    lr: float = 1e-3,
    use_batches: bool = False,
    use_cuda: bool = False,
    linear: bool = False,
    cell_offset: str = "none",
    gene_offset: str = "none",
    ldvae_bias: bool = False,
    reconstruction_loss: str = "zinb",
    dispersion: str = "gene",
    hvg_genes="all",
    point_size=10,
    dispersion_clamp=[],
    beta_disentanglement=1.0,
    kl_type="reverse",
) -> Tuple[scvi.inference.Posterior, np.ndarray]:
    """Train and return a scVI model and sample a latent space

    :param adata: sc.AnnData object non-normalized
    :param n_latent: dimension of the latent space
    :param n_epochs: number of training epochs
    :param lr: learning rate
    :param use_batches
    :param use_cuda
    :return: (scvi.Posterior, latent_space)
    """
    # Convert easily to scvi dataset
    scviDataset = AnnDatasetFromAnnData(adata)
    if isinstance(hvg_genes, int):
        scviDataset.subsample_genes(hvg_genes)

    if isinstance(scviDataset.X, np.ndarray):
        X = scviDataset.X
    else:
        X = scviDataset.X.toarray()

    # Train a model
    if not linear:
        vae = VAEGeneCell(
            scviDataset.nb_genes,
            n_batch=scviDataset.n_batches * use_batches,
            n_latent=n_latent,
            n_layers=n_encoder,
            cell_offset=cell_offset,
            gene_offset=gene_offset,
            reconstruction_loss=reconstruction_loss,
            dispersion=dispersion,
            dispersion_clamp=dispersion_clamp,
            beta_disentanglement=beta_disentanglement,
            kl_type=kl_type,
        )
    else:
        vae = LDVAEGeneCell(
            scviDataset.nb_genes,
            n_batch=scviDataset.n_batches * use_batches,
            n_latent=n_latent,
            n_layers_encoder=n_encoder,
            cell_offset=cell_offset,
            gene_offset=gene_offset,
            bias=ldvae_bias,
            reconstruction_loss=reconstruction_loss,
            dispersion=dispersion,
        )
    trainer = UnsupervisedTrainer(vae, scviDataset, train_size=1.0, use_cuda=use_cuda)
    trainer.train(n_epochs=n_epochs, lr=lr)

    # Extract latent space
    posterior = trainer.create_posterior(
        trainer.model, scviDataset, indices=np.arange(len(scviDataset))
    ).sequential()

    latent, _, _ = posterior.get_latent()

    return posterior, latent, vae, trainer


def RunVAE(
    adata,
    reconstruction_loss,
    n_latent=30,
    n_encoder=1,
    linear=False,
    cell_offset="none",
    gene_offset="none",
    ldvae=False,
    ldvae_bias=False,
    title_prefix="",
    dispersion="gene",
    hvg_genes="all",
    point_size=5,
    n_epochs=200,
    lr=1e-3,
    batch_size=1000,
    use_cuda=False,
    legend_loc="on data",
    figsize=(10, 5),
    legend_fontweight="normal",
    sct_cell_pars=None,
    outdir=None,
    sct_gene_pars=None,
    sct_model_pars_fit=None,
    dispersion_clamp=[],
    beta_disentanglement=1.0,
    kl_type="reverse",
):
    sct_gene_pars_df = pd.read_csv(sct_gene_pars, sep="\t", index_col=0)
    sct_model_pars_fit_df = pd.read_csv(sct_model_pars_fit, sep="\t", index_col=0)
    sct_model_paras_withgmean = sct_model_pars_fit_df.join(sct_gene_pars_df)

    scvi_posterior, scvi_latent, scvi_vae, scvi_trainer = compute_scvi_latent(
        adata,
        n_encoder=n_encoder,
        n_epochs=n_epochs,
        n_latent=n_latent,
        use_cuda=use_cuda,
        linear=linear,
        cell_offset=cell_offset,
        gene_offset=gene_offset,
        reconstruction_loss=reconstruction_loss,
        dispersion=dispersion,
        hvg_genes=hvg_genes,
        dispersion_clamp=dispersion_clamp,
        beta_disentanglement=beta_disentanglement,
        kl_type=kl_type,
    )

    suffix = "_{}_{}_{}_{}".format(
        cell_offset, gene_offset, reconstruction_loss, dispersion
    )
    scviDataset = AnnDatasetFromAnnData(adata)
    if isinstance(hvg_genes, int):
        scviDataset.subsample_genes(hvg_genes)

    # posterior freq of genes per cell
    # scale = scvi_posterior.sequential(batch_size=batch_size).get_sample_scale()
    # scale = scale.detach()
    scale = scvi_posterior.get_sample_scale()
    # batch_size=batch_size
    for _ in range(99):
        scale += scvi_posterior.get_sample_scale()
    scale /= 100

    scale_df = pd.DataFrame(scale)
    scale_df.index = list(adata.obs_names)
    scale_df.columns = list(scviDataset.gene_ids)
    scale_df = scale_df.T

    scvi_latent_df = pd.DataFrame(scvi_latent)
    scvi_latent_df.index = list(adata.obs_names)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        scale_df.to_csv(
            os.path.join(outdir, "SCVI_scale_df_{}.tsv".format(suffix)),
            sep="\t",
            index=True,
            header=True,
        )
        scvi_latent_df.to_csv(
            os.path.join(outdir, "SCVI_latent_df_{}.tsv".format(suffix)),
            sep="\t",
            index=True,
            header=True,
        )

    adata.obsm["X_scvi"] = scvi_latent

    for gene, gene_scale in zip(adata.var.index, np.squeeze(scale).T):
        adata.obs["scale_" + gene] = gene_scale

    sc.pp.neighbors(adata, use_rep="X_scvi", n_neighbors=20, n_pcs=30)
    sc.tl.umap(adata, min_dist=0.3)
    sc.tl.leiden(adata, key_added="X_scvi", resolution=0.8)
    X_umap = adata.obsm["X_umap"]
    X_umap_df = pd.DataFrame(X_umap)
    X_umap_df.index = list(adata.obs_names)
    if outdir:
        X_umap_df.to_csv(
            os.path.join(outdir, "SCVI_Xumap_df_{}.tsv".format(suffix)),
            sep="\t",
            index=True,
            header=True,
        )

    scviDataset = AnnDatasetFromAnnData(adata)
    if isinstance(hvg_genes, int):
        scviDataset.subsample_genes(hvg_genes)

    if isinstance(scviDataset.X, np.ndarray):
        X = scviDataset.X
    else:
        X = scviDataset.X.toarray()
    try:
        X = torch.from_numpy(X).float().to(torch.cuda.current_device())
        batch = torch.from_numpy(scviDataset.batch_indices.astype(float)).to(
            torch.cuda.current_device()
        )
    except:
        X = torch.from_numpy(X).float()
        batch = torch.from_numpy(scviDataset.batch_indices.astype(float))

    inference = scvi_vae.inference(X, batch)
    # torch.cuda.empty_cache()
    if reconstruction_loss == "nb":
        reconst_loss = log_nb_positive(
            X,
            inference["px_rate"],
            inference["px_r"],
            inference["px_dropout"],
        )
    elif reconstruction_loss == "zinb":
        reconst_loss = log_zinb_positive(
            X,
            inference["px_rate"],
            inference["px_r"],
            inference["px_dropout"],
        )

    gene_loss = np.nansum(reconst_loss.detach().cpu().numpy(), axis=0)
    cell_loss = np.nansum(reconst_loss.detach().cpu().numpy(), axis=1)

    gene_mean = np.array(adata[:, scviDataset.gene_names].X.mean(0))[0]
    if not gene_mean.shape:
        # TODO: need to handle this more gracefully
        gene_mean = np.array(adata[:, scviDataset.gene_names].X.mean(0))
    cell_mean = np.array(adata[:, scviDataset.gene_names].X.mean(1)).flatten()

    fig1 = plt.figure(figsize=figsize)
    ax = fig1.add_subplot(121)

    ax.scatter(
        gene_mean, gene_loss, label="Gene", alpha=0.5, color="black", s=point_size
    )
    gene_loss_df = pd.DataFrame([gene_mean, gene_loss])
    gene_loss_df = gene_loss_df.T
    gene_loss_df.index = list(scviDataset.gene_names)
    gene_loss_df.columns = ["gene_mean", "gene_loss"]

    cell_loss_df = pd.DataFrame([cell_mean, cell_loss])
    cell_loss_df = cell_loss_df.T
    cell_loss_df.index = list(adata.obs_names)
    cell_loss_df.columns = ["cell_mean", "cell_loss"]

    if outdir:
        gene_loss_df.to_csv(
            os.path.join(outdir, "SCVI_geneloss_df_{}.tsv".format(suffix)),
            sep="\t",
            index=True,
            header=True,
        )
        cell_loss_df.to_csv(
            os.path.join(outdir, "SCVI_cellloss_df_{}.tsv".format(suffix)),
            sep="\t",
            index=True,
            header=True,
        )
    ax.set_xlabel("Mean counts")
    ax.set_ylabel("Reconstuction loss")
    ax.legend(scatterpoints=1)

    ax = fig1.add_subplot(122)
    sc.pl.umap(
        adata,
        color="named_clusters",
        show=False,
        legend_fontweight=legend_fontweight,
        ax=ax,
        size=point_size,
        legend_loc=legend_loc,
    )
    title = "{} | Genewise | disp:{} | loss:{} | ldvae:{}({}) | n_enc:{} | c_ofst:{} | g_ofst:{}".format(
        title_prefix,
        dispersion,
        reconstruction_loss,
        ldvae,
        ldvae_bias,
        n_encoder,
        cell_offset,
        gene_offset,
    )
    fig1.suptitle(title)
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    title = title.replace(" ", "").replace("=", "_")
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        fig1.savefig(os.path.join(outdir, "{}.pdf".format(title)))
        fig1.savefig(os.path.join(outdir, "{}.png".format(title)))

    fig2 = plt.figure(figsize=figsize)
    ax = fig2.add_subplot(121)
    ax.scatter(cell_mean, cell_loss, label="Cell", alpha=0.5, s=point_size)
    ax.set_xlabel("Mean counts")
    ax.set_ylabel("Reconstuction loss")
    ax.legend(scatterpoints=1)

    ax = fig2.add_subplot(122)
    sc.pl.umap(
        adata,
        color="named_clusters",
        show=False,
        ax=ax,
        legend_loc=legend_loc,
        legend_fontweight=legend_fontweight,
        size=point_size,
    )

    title = "{} | Cellwise | disp:{} | loss:{} | ldvae:{}({}) | n_enc:{} | c_ofst:{} | g_ofst:{}".format(
        title_prefix,
        dispersion,
        reconstruction_loss,
        ldvae,
        ldvae_bias,
        n_encoder,
        cell_offset,
        gene_offset,
    )

    fig2.suptitle(title)
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    title = title.replace(" ", "").replace("=", "_")

    if outdir:
        fig2.savefig(os.path.join(outdir, "{}.pdf".format(title)))
        fig2.savefig(os.path.join(outdir, "{}.png".format(title)))

    if outdir:
        model_name = "{} | Posterior | disp:{} | loss:{} | ldvae:{}({}) | n_enc:{} | c_ofst:{} | g_ofst:{}".format(
            title_prefix,
            dispersion,
            reconstruction_loss,
            ldvae,
            ldvae_bias,
            n_encoder,
            cell_offset,
            gene_offset,
        )
        # scVI explicitly asks this path to be empty
        shutil.rmtree(
            os.path.join(outdir, model_name.replace(" ", "") + ".posterior"),
            ignore_errors=True,
        )
        scvi_posterior.save_posterior(
            os.path.join(outdir, model_name.replace(" ", "") + ".posterior")
        )

    if sct_cell_pars is None:
        fig1.show()
        fig2.show()
        obj_to_return = (
            scvi_posterior,
            scvi_latent,
            scvi_vae,
            scvi_trainer,
            fig1,
            fig2,
            None,
        )
        titles_to_return = (
            "posterior",
            "latent",
            "vae",
            "trainer",
            "cellwise_plot",
            "genewise_plot",
            "libsize_plot",
        )
        return dict(zip(titles_to_return, obj_to_return))

    title = "{} | Libsize | disp:{} | loss:{} | ldvae:{}({}) | n_enc:{} | c_ofst:{} | g_ofst:{}".format(
        title_prefix,
        dispersion,
        reconstruction_loss,
        ldvae,
        ldvae_bias,
        n_encoder,
        cell_offset,
        gene_offset,
    )
    library_sizes = pd.DataFrame(scvi_posterior.get_stats())
    sct_library_sizes = pd.read_csv(sct_cell_pars, sep="\t")
    library_sizes.index = adata.obs_names
    library_sizes.columns = ["scvi_libsize"]
    library_sizes["scvi_loglibsize"] = np.log10(library_sizes["scvi_libsize"])
    library_size_df = library_sizes.join(sct_library_sizes)

    fig3 = plt.figure(figsize=(10, 5))
    ax = fig3.add_subplot(121)
    ax.scatter(
        library_size_df["log_umi"],
        library_size_df["scvi_libsize"],
        alpha=0.5,
        s=point_size,
    )
    ax.set_xlabel("log_umi")
    ax.set_ylabel("scvi_libsize")

    ax = fig3.add_subplot(122)
    sc.pl.umap(
        adata,
        color="named_clusters",
        show=False,
        ax=ax,
        legend_fontweight=legend_fontweight,
        legend_loc=legend_loc,
        size=point_size,
    )
    fig3.suptitle(title)
    fig3.tight_layout(rect=[0, 0.03, 1, 0.95])
    title = title.replace(" ", "").replace("=", "_")
    if outdir:
        fig3.savefig(os.path.join(outdir, "{}.pdf".format(title)))
        fig3.savefig(os.path.join(outdir, "{}.png".format(title)))

    fig1.show()
    fig2.show()
    fig3.show()

    means_df = []
    dropout_df = []
    dispersion_df = []

    for tensors in scvi_posterior.sequential(batch_size=batch_size):
        sample_batch, _, _, batch_index, labels = tensors
        outputs = scvi_posterior.model.inference(
            sample_batch, batch_index=batch_index, y=labels
        )

        px_r = outputs["px_r"].detach().cpu().numpy()
        px_rate = outputs["px_rate"].detach().cpu().numpy()
        px_dropout = outputs["px_dropout"].detach().cpu().numpy()

        dropout_df.append(px_dropout)
        dispersion_df.append(px_r)
        means_df.append(px_rate)

    dropout_df = pd.DataFrame(np.vstack(dropout_df))
    dispersion_df = pd.DataFrame(np.vstack(dispersion_df))
    means_df = pd.DataFrame(np.vstack(means_df))

    means_df.index = list(adata.obs_names)
    means_df.columns = list(scviDataset.gene_names)
    means_df = means_df.T

    dropout_df.index = list(adata.obs_names)
    dropout_df.columns = list(scviDataset.gene_names)
    dropout_df = dropout_df.T

    dispersion_df.index = list(adata.obs_names)
    dispersion_df.columns = list(scviDataset.gene_names)
    dispersion_df = dispersion_df.T
    reconst_loss_df = pd.DataFrame(reconst_loss.detach().cpu().numpy())
    reconst_loss_df.index = list(adata.obs_names)
    reconst_loss_df.columns = list(scviDataset.gene_names)
    reconst_loss_df = reconst_loss_df.T
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        means_df.to_csv(
            os.path.join(outdir, "SCVI_means_df_{}.tsv".format(suffix)),
            sep="\t",
            index=True,
            header=True,
        )
        dropout_df.to_csv(
            os.path.join(outdir, "SCVI_dropout_df_{}.tsv".format(suffix)),
            sep="\t",
            index=True,
            header=True,
        )
        dispersion_df.to_csv(
            os.path.join(outdir, "SCVI_dispersions_df_{}.tsv".format(suffix)),
            sep="\t",
            index=True,
            header=True,
        )
        reconst_loss_df.to_csv(
            os.path.join(outdir, "SCVI_reconstloss_df_{}.tsv".format(suffix)),
            sep="\t",
            index=True,
            header=True,
        )

    obj_to_return = (
        scvi_posterior,
        scvi_latent,
        scvi_vae,
        scvi_trainer,
        fig1,
        fig2,
        fig3,
    )
    titles_to_return = (
        "posterior",
        "latent",
        "vae",
        "trainer",
        "cellwise_plot",
        "genewise_plot",
        "libsize_plot",
    )
    sct_gene_pars_df = pd.read_csv(sct_gene_pars, sep="\t", index_col=0)
    gene_cell_disp_summary_df = pd.DataFrame(
        dispersion_df.median(1), columns=["gene_cell_mean_disp"]
    )

    merged_df = sct_gene_pars_df.join(gene_cell_disp_summary_df).dropna()
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(121)
    ax.scatter(
        merged_df["gmean"], merged_df["gene_cell_mean_disp"], alpha=0.5, label="Gene"
    )
    ax.legend(frameon=False)
    ax.set_xlabel("Gene gmean")
    ax.set_ylabel("SCVI theta")

    merged_df = sct_gene_pars_df.join(sct_model_pars_fit_df)

    ax = fig.add_subplot(122)
    ax.scatter(merged_df["gmean"], merged_df["theta"], alpha=0.5, label="Gene")
    ax.legend(frameon=False)  # , loc='upper left')
    ax.set_xlabel("Gene gmean")
    ax.set_ylabel("SCT theta")

    title = "{} | ThetaVSGmean | disp:{} | loss:{} | ldvae:{}({}) | n_enc:{} | c_ofst:{} | g_ofst:{}".format(
        title_prefix,
        dispersion,
        reconstruction_loss,
        ldvae,
        ldvae_bias,
        n_encoder,
        cell_offset,
        gene_offset,
    )

    fig.suptitle(title)
    fig.tight_layout()
    title = title.replace(" ", "")
    if outdir:
        fig.savefig(os.path.join(outdir, "{}.pdf".format(title)))
        fig.savefig(os.path.join(outdir, "{}.png".format(title)))
    sct_library_sizes = pd.read_csv(sct_cell_pars, sep="\t")
    mean_scvi_disp_df = pd.DataFrame(dispersion_df.mean(1), columns=["scvi_dispersion"])
    sct_disp_df = pd.read_csv(
        sct_cell_pars.replace("_cell_", "_model_"), sep="\t", index_col=0
    )
    joined_df = sct_disp_df.join(mean_scvi_disp_df)

    title = "{} | Dispersion | disp:{} | loss:{} | ldvae:{}({}) | n_enc:{} | c_ofst:{} | g_ofst:{}".format(
        title_prefix,
        dispersion,
        reconstruction_loss,
        ldvae,
        ldvae_bias,
        n_encoder,
        cell_offset,
        gene_offset,
    )

    fig4 = plt.figure(figsize=(10, 5))
    ax = fig4.add_subplot(121)
    ax.scatter(joined_df["theta"], joined_df["scvi_dispersion"], alpha=0.5)
    ax.axline([0, 0], [1, 1], color="gray", linestyle="dashed")
    ax.set_xlabel("SCT theta")
    ax.set_ylabel("scVI theta")

    ax = fig4.add_subplot(122)
    sc.pl.umap(
        adata,
        color="named_clusters",
        show=False,
        ax=ax,
        legend_fontweight=legend_fontweight,
        legend_loc=legend_loc,
        size=point_size,
    )
    fig4.suptitle(title)
    fig4.tight_layout(rect=[0, 0.03, 1, 0.95])
    title = title.replace(" ", "").replace("=", "_")
    if outdir:
        fig4.savefig(os.path.join(outdir, "{}.pdf".format(title)))
        fig4.savefig(os.path.join(outdir, "{}.png".format(title)))
    return dict(zip(titles_to_return, obj_to_return))


def RunSCVI(
    counts_dir,
    metadata_file,
    sct_cell_pars,
    outdir,
    title_prefix="",
    idents_col="phenoid",
    reconstruction_loss="nb",
    dispersion="gene-cell",
    cell_offset="none",
    gene_offset="none",
    n_encoder=1,
    hvg_genes=3000,
    ldvae=False,
    ldvae_bias=False,
    use_cuda=True,
    genes_to_exclude_file=None,
    lr=1e-3,
    kl_type="reverse",
    **kwargs,
):
    adata = sc.read_10x_mtx(counts_dir)
    metadata = pd.read_csv(metadata_file, sep="\t", index_col=0)
    adata.obs["named_clusters"] = metadata[idents_col]
    n_epochs = np.min([round((20000 / adata.n_obs) * 400), 400])
    sct_model_pars_fit = sct_cell_pars.replace("cell_pars", "model_pars_fit")
    sct_gene_pars = sct_cell_pars.replace("cell_pars", "gene_attrs")

    if genes_to_exclude_file:
        genes_to_exclude_df = pd.read_csv(genes_to_exclude_file, sep="\t", index_col=0)
        genes_to_exclude = genes_to_exclude_df.index.tolist()
        all_genes = adata.var_names

        genes_to_keep = list(set(all_genes).difference(genes_to_exclude))
        adata = adata[:, genes_to_keep]

    results = RunVAE(
        adata,
        reconstruction_loss,
        linear=ldvae,
        title_prefix=title_prefix,
        n_encoder=n_encoder,
        cell_offset=cell_offset,
        gene_offset=gene_offset,
        hvg_genes=hvg_genes,
        n_epochs=n_epochs,
        lr=lr,
        dispersion=dispersion,
        use_cuda=use_cuda,
        sct_cell_pars=sct_cell_pars,
        sct_gene_pars=sct_gene_pars,
        sct_model_pars_fit=sct_model_pars_fit,
        outdir=outdir,
        kl_type=kl_type,
        **kwargs,
    )
    return results
