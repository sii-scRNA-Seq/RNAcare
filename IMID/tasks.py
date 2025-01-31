from .constants import NUMBER_CPU_LIMITS, ONTOLOGY
import scanpy as sc
import math
from matplotlib import pyplot as plt
from threadpoolctl import threadpool_limits
from sklearn.linear_model import LogisticRegressionCV
import base64
import io
import numpy as np
import scipy.stats as stats

from celery import shared_task
import pandas as pd

from .utils import (
    normalize1,
    loadSharedData,
    integrateCliData,
    integrateExData,
    getTopGeneCSV,
    clusteringPostProcess,
    go_it,
    delete_inactive_accounts,
    ICA,
)
from django.db import transaction
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap.umap_ as umap
from .models import MetaFileColumn
import hdbscan
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from sicaomics.annotate import toppfun


@shared_task
def cleanup_inactive_users():
    delete_inactive_accounts()


@shared_task
def vlnPlot(geneList, adata, groupby):
    sc.set_figure_params(dpi=100)
    sc.settings.verbosity = 0

    # Check the number of clusters
    unique_clusters = adata.obs[groupby].unique()
    # num_clusters = len(unique_clusters)

    # Prepare figure dimensions
    num_genes = len(geneList)
    max_cols = 3
    num_rows = math.ceil(num_genes / max_cols)
    fig_width = 4.5 * max_cols
    fig_height = 3 * num_rows

    # Create subplots
    fig, axes = plt.subplots(
        num_rows, max_cols, figsize=(fig_width, fig_height), sharex=False, sharey=False
    )
    axes = axes.flatten()

    # Iterate over genes and plot
    with threadpool_limits(limits=NUMBER_CPU_LIMITS, user_api="blas"):
        with plt.rc_context():
            figure1 = io.BytesIO()
            for i, gene in enumerate(geneList):
                if i < num_genes:
                    ax = axes[i]
                    sc.pl.violin(adata, [gene], groupby=groupby, ax=ax)
                    ax.set_title(gene)  # Add title to subplot

                    # Perform ANOVA (works for both 2 and >2 clusters)
                    data = [
                        adata[adata.obs[groupby] == cluster][:, gene]
                        .X.toarray()
                        .flatten()
                        for cluster in unique_clusters
                    ]
                    f_stat, p_value = stats.f_oneway(*data)
                    ax.text(
                        0.5,
                        0.95,
                        f"PValue = {p_value:.2e}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=10,
                        color="red",
                    )

            # Remove unused subplots
            for ax in axes[num_genes:]:
                fig.delaxes(ax)

            # Adjust layout and save figure
            plt.tight_layout()
            plt.savefig(figure1, format="svg", bbox_inches="tight")

    # Encode the image to return as base64
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
            plt.savefig(figure1, format="svg", bbox_inches="tight")

    image_data = base64.b64encode(figure1.getvalue()).decode("utf-8")
    return image_data


@shared_task
def heatmapPlot(geneList, adata, groupby):
    # scale and store results in layer
    figure1 = io.BytesIO()
    adata.layers["scaled"] = sc.pp.scale(adata, copy=True).X
    with threadpool_limits(limits=NUMBER_CPU_LIMITS, user_api="blas"):
        sc.pl.heatmap(
            adata,
            geneList,
            groupby=groupby,
            layer="scaled",
            vmin=-2,
            vmax=2,
            cmap="viridis",
            dendrogram=True,
            swap_axes=True,
            figsize=(11, 4),
        )
        plt.savefig(figure1, format="svg", bbox_inches="tight")

    image_data = base64.b64encode(figure1.getvalue()).decode("utf-8")
    return image_data


@shared_task(time_limit=180, soft_time_limit=150)
def runLasso(x, y, colNames, num):
    try:
        y = np.array(y)
        if num == 1:
            model = LogisticRegressionCV(
                cv=5,
                penalty="l1",
                solver="saga",
                class_weight="balanced",
                scoring="roc_auc",
                random_state=40,
                n_jobs=-1,
                max_iter=10000,
                tol=0.001,
            )
        else:
            model = LogisticRegressionCV(
                cv=5,
                penalty="l2",
                solver="saga",
                class_weight="balanced",
                scoring="roc_auc",
                random_state=40,
                n_jobs=-1,
                max_iter=10000,
                tol=0.001,
                Cs=np.logspace(-4, 4, 1000),
            )
        model.fit(x, y)
    except Exception as e:
        raise e
    coef = pd.Series(model.coef_[0], colNames).sort_values(key=abs, ascending=False)
    if len(coef[coef != 0]) == 0:
        return b""
    coef[coef != 0][:50].plot.bar(
        x="Features", y="Coef", figure=plt.figure(), fontsize=6
    )
    plt.xlabel("Features", fontsize=8)
    plt.ylabel("Coefficient Magnitude", fontsize=8)
    if num == 1:
        plt.title("Top Features Identified by Lasso Regression", fontsize=10)
    else:
        plt.title("Top Features Identified by Ridge Regression", fontsize=10)
    with io.BytesIO() as buffer:
        plt.savefig(buffer, format="svg", bbox_inches="tight")
        buffer.seek(0)
        image = buffer.getvalue()
    return base64.b64encode(image).decode("utf-8")


@shared_task(time_limit=180, soft_time_limit=150)
def runIntegrate(request, integrate, cID, log2, corrected, usr, fr):
    files, files_meta = loadSharedData(request, integrate, cID)
    temp0 = integrateCliData(request, integrate, cID, files_meta)
    if temp0.shape == (0, 0):
        raise Exception("Can't find meta file")
    if len(files) == 0:
        raise Exception("Can't find expression file")
    dfs1 = integrateExData(files, temp0, log2, corrected)
    if dfs1 is None:
        raise Exception("No matched data for meta and omics")
    # combine Ex and clinic data
    temp = dfs1.set_index("ID_REF").join(
        normalize1(temp0, log2).set_index("ID_REF"), how="inner"
    )
    temp["obs"] = temp.index.tolist()
    # temp.dropna(axis=1, inplace=True)
    usr.setIntegrationData(temp)
    X2D = runFeRed.apply(args=[fr, usr]).result
    usr.setFRData(X2D)
    usr.metagenes = pd.DataFrame()
    usr.metageneCompose = pd.DataFrame()
    if usr.save() is False:
        raise Exception("Error for creating user records.")
    try:
        with transaction.atomic():
            new_file_columns = []
            MetaFileColumn.objects.filter(user=request.user, cID=cID).delete()
            for cn in temp0.columns:
                if cn == "LABEL":
                    label = "1"
                else:
                    label = "0"
                if np.issubdtype(temp0[cn].dtype, np.number):
                    num_flag = "1"
                else:
                    num_flag = "0"
                temp_meta = MetaFileColumn(
                    user=request.user,
                    cID=cID,
                    colName=cn,
                    label=label,
                    numeric=num_flag,
                )
                if temp_meta is None:
                    raise Exception("MetaFileColumn create Failed.")
                else:
                    new_file_columns.append(temp_meta)
            MetaFileColumn.objects.bulk_create(new_file_columns)
    except Exception as e:
        raise Exception(f"Error for registering to DataBase. {str(e)}")
    return


@shared_task(time_limit=180, soft_time_limit=150)
def runDgea(clusters, adata, targetLabel, n_genes):
    try:
        if clusters == "default":
            with plt.rc_context():
                figure1 = io.BytesIO()
                figure2 = io.BytesIO()
                with threadpool_limits(limits=NUMBER_CPU_LIMITS, user_api="blas"):
                    if len(set(adata.obs["batch1"])) > 1:
                        sc.tl.rank_genes_groups(
                            adata, groupby="batch1", method="t-test"
                        )
                        sc.tl.dendrogram(adata, groupby="batch1")
                        sc.pl.rank_genes_groups_dotplot(
                            adata, n_genes=int(n_genes), show=False, color_map="bwr"
                        )
                        plt.savefig(
                            figure1,
                            format="svg",
                            bbox_inches="tight",
                        )
                        figure1 = base64.b64encode(figure1.getvalue()).decode("utf-8")
                    else:
                        figure1 = ""
                    if len(set(adata.obs[targetLabel])) > 1:
                        adata.obs[targetLabel] = adata.obs[targetLabel].astype(str)
                        sc.tl.rank_genes_groups(
                            adata, groupby=targetLabel, method="t-test"
                        )
                        sc.tl.dendrogram(adata, groupby=targetLabel)
                        sc.pl.rank_genes_groups_dotplot(
                            adata, n_genes=int(n_genes), show=False, color_map="bwr"
                        )
                        plt.savefig(
                            figure2,
                            format="svg",
                            bbox_inches="tight",
                        )
                        figure2 = base64.b64encode(figure2.getvalue()).decode("utf-8")
                    else:
                        figure2 = ""
                return [figure1, figure2]
        elif clusters == "fileName":
            return getTopGeneCSV(adata, "batch1", n_genes)
        elif clusters == "label":
            return getTopGeneCSV(adata, targetLabel, n_genes)
        elif clusters in ("LEIDEN", "HDBSCAN", "KMeans"):
            return getTopGeneCSV(adata, "cluster", n_genes)
    except:
        raise Exception("Error for running Dgea.")


@shared_task(time_limit=180, soft_time_limit=150)
def runClustering(cluster, adata, X2D, usr, param):
    if cluster == "LEIDEN":
        if param is None:
            param = 1
        try:
            param = float(param)
        except:
            raise Exception("Resolution should be a float")
        try:
            with threadpool_limits(limits=NUMBER_CPU_LIMITS, user_api="blas"):
                sc.pp.neighbors(adata, n_neighbors=40, n_pcs=40)
                sc.tl.leiden(adata, resolution=param)
        except Exception as e:
            raise Exception(f"{e}")
        Resp = clusteringPostProcess(X2D, adata, "leiden", usr)
        return Resp
    elif cluster == "HDBSCAN":
        if param is None:
            param = 20
        try:
            param = int(param)
        except:
            raise Exception("K should be positive integer.")
        if param < 5:
            raise Exception("minSize should be at least 5.")
        labels = hdbscan.HDBSCAN(min_cluster_size=int(param)).fit_predict(
            adata.obsm["X_pca"]
        )
        if min(labels) >= 0:
            labels = [str(i) for i in labels]
        else:
            labels = [str(i + 1) for i in labels]  # for outlier, will be assigned as -1

        adata.obs["hdbscan"] = labels
        adata.obs["hdbscan"] = adata.obs["hdbscan"].astype("category")
        Resp = clusteringPostProcess(X2D, adata, "hdbscan", usr)
        return Resp
    elif cluster == "Kmeans":
        try:
            param = int(param)
        except:
            raise Exception("K should be positive integer.")

        if param <= 1:
            raise Exception("K should be larger than 1.")
        km = KMeans(n_clusters=int(param), random_state=42, n_init="auto").fit(
            adata.obsm["X_pca"]
        )
        labels = [str(i) for i in km.labels_]
        adata.obs["kmeans"] = labels
        adata.obs["kmeans"] = adata.obs["kmeans"].astype("category")
        Resp = clusteringPostProcess(X2D, adata, "kmeans", usr)
        return Resp


@shared_task(time_limit=180, soft_time_limit=150)
def runGoEnrich(usr, colName, cluster_n):
    df = usr.getCorrectedCSV()
    if (
        any(df.columns.str.startswith("c_")) is True
        or len(set(df.columns).intersection({"age", "crp", "bmi", "esr", "BMI"})) > 0
    ):
        raise Exception("Not Allowed Clinic Data")
    markers = usr.getMarkers(colName)
    if markers is None:
        raise Exception("Please run clustering method first.")
    markers = markers[
        (markers.pvals_adj < 0.05)
        & (markers.logfoldchanges > 0.5)
        & (markers.group.astype(str) == str(cluster_n))
    ]
    if len(markers.index) == 0:
        raise Exception("No marker genes")
    with threadpool_limits(limits=2, user_api="blas"):
        df = go_it(markers.names.values)
    df1 = (
        df.sort_values(by=["class", "p_corr"])  # Sort by 'class' and then by 'p_corr'
        .groupby("class")  # Group by 'class'
        .head(10)  # Take the top 10 rows from each group
        .reset_index(drop=True)  # Reset the index
    )
    js = {
        "data": [
            {
                "term": row["term"],
                "class": row["class"],
                "p_corr": f"{row['p_corr']:.2e}",
                "n_genes": row["n_genes"],
                "n_genes/n_go": f"{row['n_genes/n_go']:.2e}",
                "genes": ",".join(row["genes"]),
            }
            for _, row in df1.iterrows()
        ]
    }
    return js

    # fig = go.Figure()

    # def fig_add_trace_ontology(fig, df, ontology):
    #     fig.add_trace(
    #         go.Bar(
    #             y=df[df["class"] == ontology.name].term[::-1],
    #             x=df[df["class"] == ontology.name].per[::-1],
    #             name=ontology.name,
    #             customdata=[
    #                 "P_corr=" + str(round(i, 5))
    #                 for i in df[df["class"] == ontology.name].p_corr[::-1]
    #             ],
    #             hovertemplate="Ratio: %{x:.5f}<br> %{customdata}",
    #             orientation="h",
    #             marker={
    #                 "color": df[df["class"] == ontology.name].p_corr[::-1],
    #                 "colorscale": ontology.color,
    #             },
    #         )
    #     )

    # for o in ONTOLOGY.values():
    #     fig_add_trace_ontology(fig, df1, o)

    # fig.update_layout(
    #     barmode="stack",
    #     height=1200,
    #     uniformtext_minsize=50,
    #     uniformtext_mode="hide",
    #     xaxis=dict(title="Gene Ratio"),
    # )
    # return base64.b64encode(fig.to_image(format="svg")).decode("utf-8")


@shared_task(time_limit=180, soft_time_limit=150)
def runFeRed(fr, usr):
    pca_temp = usr.getAnndata().obsm["X_pca"]
    if fr == "TSNE":
        tsne = TSNE(
            n_components=2,
            random_state=42,
            n_jobs=2,
            perplexity=min(30.0, pca_temp.shape[0] - 1),
        )
        X2D = tsne.fit_transform(pca_temp)
        usr.redMethod = "TSNE"
    elif fr == "UMAP":
        umap1 = umap.UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=min(30, pca_temp.shape[0] // 2),
            metric="euclidean",
            init="spectral",
            n_epochs=None,
            learning_rate=1.0,
            n_jobs=2,
            spread=1.0,
            min_dist=0.1,
        )
        X2D = umap1.fit_transform(pca_temp)
        usr.redMethod = "UMAP"
    else:
        pca = PCA(n_components=2, random_state=42)
        X2D = pca.fit_transform(pca_temp)
        usr.redMethod = "PC"
    return X2D


@shared_task(time_limit=180, soft_time_limit=150)
def runICA(df, num):
    df = df.copy()
    df.index.name = "ID_REF"
    # Drop the columns 'FileName', 'obs', and 'LABEL'
    df.drop(columns=["FileName", "obs", "LABEL"], inplace=True)

    # Separate the columns
    # Columns not starting with 'c_'
    df1 = df.loc[:, ~df.columns.str.startswith("c_")]

    try:
        metagenes, metageneCompose = ICA(df1, num)
    except Exception as e:
        raise e
    return metagenes, metageneCompose


@shared_task(time_limit=180, soft_time_limit=150)
def runTopFun(metageneCompose, num):
    Tannot = toppfun.ToppFunAnalysis(
        data=metageneCompose,
        threshold=3,
        method="std",
        tail="heaviest",
        convert_ids=True,
    )
    df = Tannot.get_analysis(metagene=("metagene_" + str(num)))
    filtered_df = df[df["Category"] == "Pathway"]
    jd = {
        "data": [
            {
                "Category": row["Category"],
                "ID": row["ID"],
                "Name": row["Name"],
                "PValue": f"{row['PValue']:.2e}",
                "Gene_Symbol": row["Gene_Symbol"],
            }
            for _, row in filtered_df.iterrows()
        ]
    }
    return jd
