from .constants import NUMBER_CPU_LIMITS
import scanpy as sc
import math
from matplotlib import pyplot as plt
from threadpoolctl import threadpool_limits
from sklearn.linear_model import LassoCV
import base64
import io

from celery import shared_task
import pandas as pd


@shared_task
def vlnPlot(geneList, adata):
    sc.set_figure_params(dpi=100)
    sc.settings.verbosity = 0
    num_genes = len(geneList)
    max_cols = 3
    num_rows = math.ceil(num_genes / max_cols)
    # Set the figure size based on number of subplots
    fig_width = 4.5 * max_cols
    fig_height = 3 * num_rows
    # Create subplots
    fig, axes = plt.subplots(
        num_rows, max_cols, figsize=(fig_width, fig_height), sharex=False, sharey=False
    )
    # Flatten the axes array for easier indexing
    axes = axes.flatten()
    # Iterate over genes and plot
    with threadpool_limits(limits=NUMBER_CPU_LIMITS, user_api="blas"):
        with plt.rc_context():
            figure1 = io.BytesIO()
            for i, gene in enumerate(geneList):
                if i < num_genes:
                    ax = axes[i]
                    sc.pl.violin(adata, [gene], groupby="cluster", ax=axes[i])
                    ax.set_title(gene)  # Add title to subplot

            # Remove any unused subplots
            for ax in axes[num_genes:]:
                fig.delaxes(ax)

            # Adjust layout
            plt.tight_layout()
            plt.savefig(figure1, format="png", bbox_inches="tight")

    image_data = base64.b64encode(figure1.getvalue()).decode("utf-8")
    return image_data


@shared_task
def densiPlot(geneList, adata):
    sc.set_figure_params(dpi=100)
    sc.settings.verbosity = 0
    # Iterate over genes and plot
    with threadpool_limits(limits=NUMBER_CPU_LIMITS, user_api="blas"):
        with plt.rc_context({"figure.figsize": (4, 4)}):
            figure1 = io.BytesIO()
            sc.pl.umap(
                adata,
                color=geneList,
                s=50,
                frameon=False,
                ncols=4,
                vmax="p99",
                cmap="coolwarm",
            )
            plt.savefig(figure1, format="png", bbox_inches="tight")

    image_data = base64.b64encode(figure1.getvalue()).decode("utf-8")
    return image_data


@shared_task
def heatmapPlot(geneList, adata):
    # scale and store results in layer
    figure1 = io.BytesIO()
    adata.layers["scaled"] = sc.pp.scale(adata, copy=True).X
    with threadpool_limits(limits=NUMBER_CPU_LIMITS, user_api="blas"):
        sc.pl.heatmap(
            adata,
            geneList,
            groupby="cluster",
            layer="scaled",
            vmin=-2,
            vmax=2,
            cmap="viridis",
            dendrogram=True,
            swap_axes=True,
            figsize=(11, 4),
        )
        plt.savefig(figure1, format="png", bbox_inches="tight")

    image_data = base64.b64encode(figure1.getvalue()).decode("utf-8")
    return image_data


@shared_task(time_limit=180, soft_time_limit=150)
def runLasso(x, y):
    try:
        model = LassoCV(cv=5, random_state=42, n_jobs=-1, max_iter=10000, tol=0.01)
        model.fit(x, y)
    except Exception as e:
        raise e
    coef = pd.Series(
        model.coef_, df.drop([colName], axis=1, inplace=False).columns
    ).sort_values(key=abs, ascending=False)

    coef[coef != 0][:50].plot.bar(
        x="Features", y="Coef", figure=plt.figure(), fontsize=6
    )
    with io.BytesIO() as buffer:
        plt.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        image = buffer.getvalue()
    return image
