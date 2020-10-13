import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from .statistics import concordance_correlation_coefficient


def count_frequency_plot(gene, sampled_counts_mtx, exclude_zeroes=False, kde=False):
    sampled_counts = pd.DataFrame(sampled_counts_mtx[gene].value_counts()).rename(
        columns={gene: "frequency"}
    )
    # sampled_counts
    gene_counts = pd.Series(adata[:, gene].X.toarray().flatten())
    gene_hist = pd.DataFrame(gene_counts.value_counts()).rename(
        columns={0: "frequency"}
    )

    kstat = stats.kstest(sampled_counts_mtx[gene], gene_counts)

    fig, ax = plt.subplots()
    if kde:
        sampled_counts = pd.DataFrame(sampled_counts_mtx[gene])
        sampled_counts.columns = ["counts"]
        sampled_counts["type"] = "Generated"
        gene_counts = pd.DataFrame(adata[:, gene].X.toarray().flatten())
        gene_counts.columns = ["counts"]
        gene_counts["type"] = "Observed"

        merged_df = pd.concat([gene_counts, sampled_counts])
        if exclude_zeroes:
            merged_df = merged_df.loc[merged_df.counts > 0]
        sns.kdeplot(
            data=merged_df,
            x="counts",
            hue="type",
            palette={"Generated": "#4dac26", "Observed": "#ca0020"},
            cut=0,
            ax=ax,
            legend=True,
        )

        ax.get_legend().set_title("")
        ax.get_legend().get_frame().set_alpha(0)

    else:
        ax.scatter(
            gene_hist.index,
            gene_hist["frequency"],
            color="#ca0020",
            label="Observed",
            alpha=0.5,
            s=6,
        )
        ax.scatter(
            sampled_counts.index,
            sampled_counts["frequency"],
            color="#4dac26",
            label="Generated",
            alpha=0.5,
            s=6,
        )
        ax.legend(frameon=False)

    ax.set_xlabel("Count")
    ax.set_ylabel("Frequency")

    ax.set_title(
        gene + " | KS={} (p={})".format(np.round(kstat[0], 2), np.round(kstat[1], 2))
    )
    fig.tight_layout()
    return fig


def mean_var_plot(
    all_gene_counts,
    sampled_counts_mtx,
    clip_upper_quantiles=True,
    mean_lim=None,
    var_lim=None,
):
    fig = plt.figure(figsize=(9, 3.5))
    #             palette={"Generated": "#4dac26", "Observed": "#ca0020"},
    ax = fig.add_subplot(131)
    ax.scatter(
        all_gene_counts.mean(0),
        sampled_counts_mtx.mean(0),
        alpha=0.5,
    )
    cc = concordance_correlation_coefficient(
        all_gene_counts.mean(0), sampled_counts_mtx.mean(0)
    )
    print(cc)
    ax.axline([0, 0], [1, 1], linestyle="dashed", color="gray")
    if mean_lim:
        ax.set_ylim(0, mean_lim)
    elif clip_upper_quantiles:
        ylim = np.quantile(sampled_counts_mtx.mean(0), q=0.997)
        ax.set_ylim(0, ylim)

    ax.set_xlabel("Observed mean")
    ax.set_ylabel("Generated mean")
    ax.set_title("Mean (ccc={})".format(np.round(cc, 3)))
    if cc < 0.01:
        ax.set_title("Mean (ccc={:0.2e})".format(cc))

    ax = fig.add_subplot(132)
    ax.scatter(
        all_gene_counts.var(0),
        sampled_counts_mtx.var(0),
        alpha=0.5,
    )
    ax.axline([0, 0], [1, 1], linestyle="dashed", color="gray")
    if var_lim:
        ax.set_ylim(0, var_lim)
    elif clip_upper_quantiles:
        ylim = np.quantile(sampled_counts_mtx.var(0), q=0.997)
        ax.set_ylim(0, ylim)
    cc = concordance_correlation_coefficient(
        all_gene_counts.var(0), sampled_counts_mtx.var(0)
    )
    print(cc)
    ax.set_xlabel("Observed var")
    ax.set_ylabel("Generated var")
    ax.set_title("Variance (ccc={})".format(np.round(cc, 3)))

    if cc < 0.01:
        ax.set_title("Variance (ccc={:0.2e})".format(cc))

    ax = fig.add_subplot(133)
    ax.scatter(
        all_gene_counts.mean(0),
        all_gene_counts.var(0),
        color="#ca0020",
        label="Observed",
        alpha=0.5,
    )

    ax.scatter(
        sampled_counts_mtx.mean(0),
        sampled_counts_mtx.var(0),
        color="#4dac26",
        label="Generated",
        alpha=0.5,
    )
    if mean_lim and var_lim:
        ax.set_xlim(0, mean_lim)
        ax.set_ylim(0, var_lim)

    elif clip_upper_quantiles:
        xlim = np.quantile(sampled_counts_mtx.mean(0), q=0.997)
        ax.set_xlim(0, xlim)
        ylim = np.quantile(sampled_counts_mtx.var(0), q=0.997)
        ax.set_ylim(0, ylim)
    ax.set_xlabel("Mean")
    ax.set_ylabel("Var")
    ax.set_title("Mean vs var")
    ax.legend()

    ax.get_legend().set_title("")
    ax.get_legend().get_frame().set_alpha(0)
    fig.tight_layout()
    return fig
