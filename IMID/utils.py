from functools import lru_cache

from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.base import download_go_basic_obo
from goatools.base import download_ncbi_associations
from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS
from goatools.obo_parser import GODag

from genes_ncbi_proteincoding import GENEID2NT

import random
import string
from .constants import BASE_UPLOAD, BASE_STATIC, GeneID_URL
from .models import Gene, GOTerm, userData
from harmony import harmonize
import pandas as pd
import numpy as np
import scanpy as sc
import collections
from combat.pycombat import pycombat
from matplotlib import pyplot as plt
from django.http import HttpResponse, JsonResponse
from collections import Counter
import math
import requests
import json


@lru_cache(maxsize=None)
def build_GOEnrichmentStudyNS():
    obo_fname = download_go_basic_obo()
    godag = GODag(obo_fname)
    gene2go_fname = download_ncbi_associations()
    gene2go_reader = Gene2GoReader(gene2go_fname, taxids=[9606])
    ns2assoc = gene2go_reader.get_ns2assc()  # bp,cc,mf
    return GOEnrichmentStudyNS(
        GENEID2NT.keys(),
        ns2assoc,
        godag,
        propagate_counts=False,
        alpha=0.05,
        methods=["fdr_bh"],
    )


def zip_for_vis(X2D, batch, obs):
    traces = {}
    for i, j, k in zip(X2D, batch, obs):
        j = str(j)
        if j not in traces:
            traces[j] = {"data": [i], "obs": [k]}
        else:
            traces[j]["data"].append(i)
            traces[j]["obs"].append(k)
    return traces


def fromPdtoSangkey(df):
    df = df.copy()
    source = []
    df[["parent_level"]] = [i + " " for i in df[["LABEL"]].values]
    columns = df.columns.tolist()

    for i in columns:
        source.extend([j[0] for j in df[[i]].values])
    nodes = collections.Counter(source)

    source = {}
    for i in range(1, len(columns)):
        agg = df.groupby([columns[i - 1], columns[i]]).size().reset_index()
        agg.columns = [columns[i - 1], columns[i], "count"]
        for index, row in agg.iterrows():
            source[(row[columns[i - 1]], row[columns[i]])] = row["count"]

    result = {}
    result1 = []
    dic_node = {}
    for i, j in enumerate(nodes.keys()):
        result1.append({"node": i, "name": j})
        dic_node[j] = i  # name->number
    result["nodes"] = result1

    result1 = []
    for i in source:
        result1.append(
            {"source": dic_node[i[0]], "target": dic_node[i[1]], "value": source[i]}
        )
    result["links"] = result1
    # return json.dumps(result)

    sou = []
    tar = []
    value = []
    result = {}
    for i in result1:
        sou.append(i["source"])
        tar.append(i["target"])
        value.append(i["value"])
    result["source1"] = sou
    result["target1"] = tar
    result["value1"] = value
    result["label"] = list(nodes.keys())
    return result


def go_it(test_genes):
    goeaobj = build_GOEnrichmentStudyNS()
    goea_results_all = goeaobj.run_study(
        [g.id for g in Gene.objects.filter(name__in=test_genes)]
    )
    goea_result_sig = [
        r for r in goea_results_all if r.p_fdr_bh < 0.05 and r.ratio_in_study[0] > 1
    ]
    go_df = pd.DataFrame(
        data=map(
            lambda x: [
                x.goterm.name,
                x.goterm.namespace,
                x.p_fdr_bh,
                x.ratio_in_study[0],
                GOTerm.objects.get(name=x.GO).gene.count(),
            ],
            goea_result_sig,
        ),
        columns=[
            "term",
            "class",
            "p_corr",
            "n_genes",
            "n_go",
        ],
        index=map(lambda x: x.GO, goea_result_sig),
    )
    go_df["per"] = go_df.n_genes / go_df.n_go
    return go_df


def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = "".join(random.choice(letters) for i in range(length))
    return result_str


def handle_uploaded_file1(f, username, filename=""):
    if filename == "":
        with open(BASE_UPLOAD + username + "_" + f.name, "wb+") as dest:
            for chunk in f.chunks():
                dest.write(chunk)
    else:
        with open(BASE_UPLOAD + username + "_" + filename + ".csv", "wb+") as dest:
            for chunk in f.chunks():
                dest.write(chunk)


def combat(dfs):
    df_exp = pd.concat(dfs, join="inner", axis=1)
    batch = []
    datasets = dfs
    for j in range(len(datasets)):
        batch.extend([j for _ in range(len(datasets[j].columns))])
    df_corrected = pycombat(df_exp, batch)
    Xc = df_corrected.T
    return Xc.reset_index(drop=True)


# pip install harmony-pytorch
def harmony(dfs, batch, obs):
    dfs1 = pd.concat(dfs, join="inner", axis=0)
    adata = sc.AnnData(
        np.zeros(dfs1.values.shape), dtype=np.float64
    )  #'obs','FileName', 'LABEL'
    adata.X = dfs1.values
    adata.var_names = dfs1.columns.tolist()
    adata.obs_names = obs
    adata.obs["batch"] = batch
    # sc.tl.pca(adata)
    # sce.pp.harmony_integrate(adata, 'batch')
    # ha=adata.obsm['X_pca_harmony']
    # return pd.DataFrame(ha,columns=['PCA_'+str(i+1) for i in range(ha.shape[1])]).reset_index(drop=True)
    ha = harmonize(adata.X, adata.obs, batch_key="batch")
    return pd.DataFrame(ha, columns=dfs1.columns).reset_index(drop=True)


def bbknn(dfs):
    return dfs


def clusteringPostProcess(X2D, adata, method, BASE_STATIC, username, random_str, usr):
    if method != "kmeans" and len(set(adata.obs[method])) == 1:
        # throw error for just 1 cluster
        return HttpResponse("Only 1 Cluster after clustering", status=400)

    li = adata.obs[method].tolist()
    count_dict = Counter(li)
    for member, count in count_dict.items():
        if count < 10:
            return HttpResponse(
                "The number of data in the cluster "
                + str(member)
                + " is less than 10, which will not be able for further analysis.",
                status=405,
            )

    traces = zip_for_vis(X2D, list(adata.obs[method]), adata.obs_names.tolist())

    # adata.write(BASE_STATIC + username + "_adata.h5ad")
    adata.obs["cluster"] = li
    usr.setAnndata(adata)

    barChart1 = []
    barChart2 = []

    with plt.rc_context():
        sc.tl.rank_genes_groups(adata, groupby=method, method="t-test")
        sc.tl.dendrogram(adata, groupby=method)
        sc.pl.rank_genes_groups_dotplot(adata, n_genes=4, show=False, color_map="bwr")
        plt.savefig(
            BASE_STATIC + username + "_cluster_" + random_str + "_1.png",
            bbox_inches="tight",
        )
        sc.pl.rank_genes_groups(adata, n_genes=20, sharey=False)
        plt.savefig(
            BASE_STATIC + username + "_cluster_" + random_str + "_2.png",
            bbox_inches="tight",
        )
        markers = sc.get.rank_genes_groups_df(adata, None)
        usr.setMarkers(markers)
        b = (
            adata.obs.sort_values(["batch1", method])
            .groupby(["batch1", method])
            .count()
            .reset_index()
        )

        b = b[["batch1", method, "batch2"]]
        b.columns = ["batch", method, "count"]
        barChart1 = [
            {
                "x": sorted(list(set(b[method].tolist()))),
                "y": b[b["batch"] == i]["count"].tolist(),
                "name": i,
                "type": "bar",
            }
            for i in set(b["batch"].tolist())
        ]

        b = (
            adata.obs.sort_values(["batch2", method])
            .groupby(["batch2", method])
            .count()
            .reset_index()
        )
        b = b[["batch2", method, "batch1"]]
        b.columns = ["batch", "cluster", "count"]
        barChart2 = [
            {
                "x": sorted(list(set(b["cluster"].tolist()))),
                "y": b[b["batch"] == i]["count"].tolist(),
                "name": i,
                "type": "bar",
            }
            for i in set(b["batch"].tolist())
        ]

        usr.save()

        return JsonResponse(
            {
                "traces": traces,
                "fileName": username + "_cluster_" + random_str + "_1.png",
                "fileName1": username + "_cluster_" + random_str + "_2.png",
                "bc1": barChart1,
                "bc2": barChart2,
            }
        )


def getTopGeneCSV(adata, groupby, n_genes):
    if len(set(adata.obs[groupby])) > 1:
        sc.tl.rank_genes_groups(adata, groupby=groupby, method="t-test")
        result = adata.uns["rank_genes_groups"]
        groups = result["names"].dtype.names
        result_df = pd.DataFrame()
        for group in groups:
            top_genes = pd.DataFrame(
                result["names"][group], columns=[f"TopGene_{group}"]
            )
            result_df = pd.concat([result_df, top_genes], axis=1).loc[: int(n_genes),]
        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = "attachment; filename=topGenes.csv"
        result_df.to_csv(path_or_buf=response)
        return response

    else:
        return HttpResponse("Only one cluster", status=500)


def vlnPlot(geneList, adata, clientID, username):
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
    with plt.rc_context():
        for i, gene in enumerate(geneList):
            if i < num_genes:
                sc.pl.violin(adata, [gene], groupby="cluster", ax=axes[i])

        # Remove any unused subplots
        for ax in axes[num_genes:]:
            fig.delaxes(ax)

        # Adjust layout
        plt.tight_layout()
        plt.savefig(
            BASE_STATIC + username + "_" + clientID + "_violin.png", bbox_inches="tight"
        )
    return JsonResponse({"fileName": username + "_" + clientID + "_violin.png"})


def densiPlot(geneList, adata, clientID, username):
    sc.set_figure_params(dpi=100)
    sc.settings.verbosity = 0
    # Iterate over genes and plot
    with plt.rc_context({"figure.figsize": (4, 4)}):
        sc.pl.umap(
            adata,
            color=geneList,
            s=50,
            frameon=False,
            ncols=4,
            vmax="p99",
            cmap="coolwarm",
        )
        plt.savefig(
            BASE_STATIC + username + "_" + clientID + "_featurePlot.png",
            bbox_inches="tight",
        )

    return JsonResponse({"fileName": username + "_" + clientID + "_featurePlot.png"})


def heatmapPlot(geneList, adata, clientID, username):
    # scale and store results in layer
    adata.layers["scaled"] = sc.pp.scale(adata, copy=True).X
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
    plt.savefig(
        BASE_STATIC + username + "_" + clientID + "_heatmapPlot.png",
        bbox_inches="tight",
    )
    return JsonResponse({"fileName": username + "_" + clientID + "_heatmapPlot.png"})


def GeneID2SymID(geneList):
    geneIDs = []
    for g in geneList:
        if g.startswith("ENSG"):
            geneIDs.append(g)
    if len(geneIDs) == 0:
        return geneList
    ids_json = json.dumps(geneIDs)
    body = {"api": 1, "ids": ids_json}
    try:
        response = requests.post(GeneID_URL, data=body)
        text = json.loads(response.text)
    except:
        return None
    retRes = []
    for i in geneList:
        if i in text:
            retRes.append(text[i])
        else:
            retRes.append(i)
    return retRes


def usrCheck(request):
    username = request.user.username
    clientID = request.GET.get("cID", None)
    if clientID is None:
        return {"status": 0, "message": "clientID is Required."}
    usr = userData.read(username, clientID)
    if usr is None:
        return {
            "status": 0,
            "message": "Can't find the user/device.Please request from the beginning.",
        }
    return {"status": 1, "usrData": usr}
