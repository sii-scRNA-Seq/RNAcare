from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.contrib.auth.decorators import login_required
import pandas as pd
import json
import os, glob
from sklearn.manifold import TSNE
import umap.umap_ as umap
import scanpy as sc
import numpy as np
from matplotlib import pyplot as plt
import hdbscan

from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# gene_result.txt, genes_ncbi_proteincoding.py, go-basic.obo

import matplotlib
import re

matplotlib.use("agg")
import plotly.graph_objects as go

import requests
from bs4 import BeautifulSoup
from .constants import ONTOLOGY, BASE_UPLOAD, BASE_STATIC
from .utils import zip_for_vis, fromPdtoSangkey, go_it, get_random_string, handle_uploaded_file1, combat, harmony, bbknn



# lasso.R for data visualization
# sys.path.append('/home/mt229a/Downloads/')#gene_result.txt, genes_ncbi_proteincoding.py, go-basic.obo


@login_required()
def index(request):
    return render(
        request,
        "index.html",
    )


@login_required()
def tab(request):
    return render(
        request,
        "tab1.html",
    )


@login_required()
def uploadExpression(request):
    if request.method != "POST":
        return HttpResponse("Method not allowed", status=405)
    username = request.user.username
    for file in glob.glob(BASE_UPLOAD + "/" + username + "*"):
        os.remove(file)
    fileNames = []
    context = {}
    files = request.FILES.getlist("files[]", None)
    for f in files:
        handle_uploaded_file1(f, username)
        fileNames.append(BASE_UPLOAD + username + "_" + f.name)
        context[f.name] = {}

    for f in fileNames:
        df = pd.read_csv(f).head()
        if len(df.columns) > 7:  # only show 7 columns
            df = pd.concat([df.iloc[:, 0:4], df.iloc[:, -3:]], axis=1)

            temp = df.columns[3]
            df.rename(columns={temp: "..."}, inplace=True)
            df["..."] = "..."
        json_re = df.reset_index().to_json(orient="records")
        data = json.loads(json_re)
        context["_".join(f.split("_")[1:])]["d"] = data
        context["_".join(f.split("_")[1:])]["names"] = ["index"] + df.columns.to_list()
    return render(request, "table.html", {"root": context})

@login_required()
def uploadMeta(request):
    if request.method != "POST":
        return HttpResponse("Method not allowed", status=405)
    username = request.user.username
    files = request.FILES.getlist("meta", None)[0]
    handle_uploaded_file1(files, username, "meta")
    f = BASE_UPLOAD + username + "_meta.csv"
    context = {}
    context["metaFile"] = {}
    df = pd.read_csv(f).head()
    if len(df.columns) > 7:  # only show 7 columns
        df = pd.concat([df.iloc[:, 0:4], df.iloc[:, -3:]], axis=1)

        temp = df.columns[3]
        df.rename(columns={temp: "..."}, inplace=True)
        df["..."] = "..."
    json_re = df.reset_index().to_json(orient="records")
    data = json.loads(json_re)
    context["metaFile"]["d"] = data
    context["metaFile"]["names"] = ["index"] + df.columns.to_list()
    return render(request, "table.html", {"root": context})


@login_required()
def eda(request):
    username = request.user.username
    corrected = request.GET.get("correct", "Combat")
    log2 = request.GET.get("log2", "No")
    fr = request.GET.get("fr", "TSNE")
    integrate = request.GET.get("integrate", "").split(",")
    if "" in integrate:
        integrate.remove("")
    for file in glob.glob(BASE_STATIC + "/" + username + "*"):
        os.remove(file)

    directory = os.listdir(BASE_UPLOAD)
    files = [i for i in directory if username == i.split("_")[0]]
    files_meta=set()

    in_ta = {
        "SERA": "share_SERA_BLOOD.csv",
        "PEAC": "share_PEAC_recon.csv",
        "PSORT": "share_PSORT.csv",
        "RAMAP": "share_RAMAP_WHL.csv",
        "ORBIT":"share_ORBIT.csv"
    }
    in_ta1={
        'SERA':'share_IMID_meta.csv',
        'PEAC':'share_IMID_meta.csv',
        'PSORT':'share_IMID_meta.csv',
        'RAMAP':'share_IMID_meta.csv',
        'ORBIT':'share_ORBIT_meta.csv'
    }
    for i in integrate:
        if i in in_ta:
            files.append(in_ta[i])
            files_meta.add(in_ta1[i])
    dfs = []
    batch = []
    obs = []
    temp0 = []
    color2 = []
    flag = 0

    temp0 = pd.read_csv(BASE_UPLOAD + username + "_meta.csv")
    temp0 = temp0.dropna(axis=1)
    if len(integrate) != 0 and integrate[0] != "null":  # jquery plugin compatible
        for i in files_meta:
            temp0 = pd.concat(
                [temp0, pd.read_csv(BASE_UPLOAD + i)], axis=0, join="inner"
                )
    for file in files:
        if "meta" in file:
            flag = 1
            continue
        temp01 = pd.read_csv(BASE_UPLOAD + file).set_index("ID_REF")
        if "raw" in file or "RAW" in file:
            temp01 = temp01.div(temp01.sum(axis=1), axis=0) * 1e6
            # temp01=temp01/temp01.sum()*1e6
        temp = temp01.join(temp0[["ID_REF", "LABEL"]].set_index("ID_REF"), how="inner")
        temp1 = temp.reset_index().drop(
            ["ID_REF", "LABEL"], axis=1, inplace=False
        )  # exclude ID_REF & LABEL
        # temp0.append(temp1.filter(regex='c_'))#clinic data
        # temp1=temp1.loc[:,~temp1.columns.str.startswith('c_')]#transcript data
        # rpkm to CPM
        if temp1.shape[0] != 0:
            temp1 = (temp1.div(temp1.sum(axis=0), axis=1) * 1e6) / temp1.sum().sum()
            # exclude NA
            temp1 = temp1.dropna(axis=1)
            dfs.append(temp1)

            # color2.extend(list(temp.LABEL))
            batch.extend(
                ["_".join(file.split("_")[1:]).split(".csv")[0]] * temp1.shape[0]
            )
            # obs.extend(temp.ID_REF.tolist())
            obs.extend(temp.index.tolist())

    if log2 == "Yes":
        dfs = [np.log2(i + 1) for i in dfs]

    dfs1 = None
    if len(dfs) > 1:
        if corrected == "Combat":
            dfs1 = combat([i.T for i in dfs])
        elif corrected == "Harmony":
            dfs1 = harmony(dfs, batch, obs)
        elif corrected == "BBKNN":
            dfs1 = bbknn(dfs)
    elif len(dfs)==0:
        return HttpResponse("No matched data for meta and omics", status=400)
    else:
        dfs1 = dfs[0]
    dfs1["ID_REF"] = obs
    dfs1["FileName"] = batch
    # temp0=pd.concat(temp0,axis=0).reset_index(drop=True) #combine all clinic data
    if flag == 0:
        return HttpResponse("Can't find meta file", status=400)
    temp = dfs1.set_index("ID_REF").join(temp0.set_index("ID_REF"), how="inner")
    temp["obs"] = temp.index.tolist()
    # temp['FileName']=batch#inner join may not match so valued beforehand
    temp.to_csv(BASE_STATIC + username + "_corrected.csv", index=False)

    color2 = [i + "(" + j + ")" for i, j in zip(temp.LABEL, temp.FileName)]

    dfs1.drop(["ID_REF"], axis=1, inplace=True)
    dfs1.drop(["FileName"], axis=1, inplace=True)

    if fr == "TSNE":
        tsne = TSNE(n_components=3, random_state=42)
        X3D1 = tsne.fit_transform(dfs1)
    else:

        umap1 = umap.UMAP(n_components=3, random_state=42, n_neighbors=30)
        X3D1 = umap1.fit_transform(dfs1)

    with open(BASE_STATIC + username + "_fr.json", "w") as f:
        f.write(json.dumps(X3D1.tolist()))

    traces = zip_for_vis(X3D1.tolist(), temp.FileName, temp.obs)
    traces1 = zip_for_vis(X3D1.tolist(), color2, temp.obs)
    context = {
        "dfs1": json.dumps(traces),
        "dfs2": json.dumps(traces1),
        "fr": fr,
        "log": log2,
        "correct": corrected,
    }
    # return render(request,'eda.html',context)
    return JsonResponse(context)



@login_required()
def dgea(request):
    username = request.user.username
    clusters = request.GET.get("clusters", "default")
    n_genes = request.GET.get("topN", 4)
    df = pd.read_csv(BASE_STATIC + username + "_corrected.csv")
    t = df.loc[
        :, ~(df.columns.isin(["obs", "FileName", "LABEL"]))
    ]  #'obs','FileName', 'LABEL'
    adata = sc.AnnData(np.zeros(t.values.shape), dtype=np.float64)
    adata.X = t.values
    adata.var_names = t.columns.tolist()
    adata.obs_names = df.obs.tolist()
    adata.obs["batch1"] = df.FileName.tolist()
    adata.obs["batch2"] = [
        i + "(" + j + ")" for i, j in zip(df.LABEL.tolist(), df.FileName.tolist())
    ]
    sc.tl.pca(adata, svd_solver="arpack")
    adata.write(BASE_STATIC + username + "_adata.h5ad")
    random_str = get_random_string(8)
    if clusters == "default":
        with plt.rc_context():
            if len(set(adata.obs["batch1"])) > 1:
                sc.tl.rank_genes_groups(adata, groupby="batch1", method="t-test")
                sc.tl.dendrogram(adata, groupby="batch1")
                sc.pl.rank_genes_groups_dotplot(
                    adata, n_genes=int(n_genes), show=False, color_map="bwr"
                )
                plt.savefig(
                    BASE_STATIC + username + "_batch1_" + random_str + ".png",
                    bbox_inches="tight",
                )
            else:
                pass
            if len(set(adata.obs["batch2"])) > 1:
                sc.tl.rank_genes_groups(adata, groupby="batch2", method="t-test")
                sc.tl.dendrogram(adata, groupby="batch2")
                sc.pl.rank_genes_groups_dotplot(
                    adata, n_genes=int(n_genes), show=False, color_map="bwr"
                )
                plt.savefig(
                    BASE_STATIC + username + "_batch2_" + random_str + ".png",
                    bbox_inches="tight",
                )
            else:
                pass
        return JsonResponse(
            [
                username + "_batch1_" + random_str + ".png",
                username + "_batch2_" + random_str + ".png",
            ],
            safe=False,
        )
    if clusters == "fileName":
        # show top gene for specific group
        if len(set(adata.obs["batch1"])) > 1:
            sc.tl.rank_genes_groups(adata, groupby="batch1", method="t-test")
            result = adata.uns["rank_genes_groups"]
            groups = result["names"].dtype.names
            result_df = pd.DataFrame()
            for group in groups:
                top_genes = pd.DataFrame(
                    result["names"][group], columns=[f"TopGene_{group}"]
                )
                result_df = pd.concat([result_df, top_genes], axis=1).loc[
                    : int(n_genes),
                ]
            response = HttpResponse(content_type="text/csv")
            response["Content-Disposition"] = "attachment; filename=topGenes.csv"
            result_df.to_csv(path_or_buf=response)
            return response

        else:
            pass
    elif clusters == "label":
        if len(set(adata.obs["batch2"])) > 1:
            sc.tl.rank_genes_groups(adata, groupby="batch2", method="t-test")
            result = adata.uns["rank_genes_groups"]
            groups = result["names"].dtype.names
            result_df = pd.DataFrame()
            print(groups)
            for group in groups:
                top_genes = pd.DataFrame(
                    result["names"][group], columns=[f"TopGene_{group}"]
                )
                result_df = pd.concat([result_df, top_genes], axis=1).loc[
                    : int(n_genes),
                ]
            response = HttpResponse(content_type="text/csv")
            response["Content-Disposition"] = "attachment; filename=topGenes.csv"
            result_df.to_csv(path_or_buf=response)
            return response

        else:
            pass


@login_required()
def clustering(request):
    username = request.user.username
    cluster = request.GET.get("cluster", "LEIDEN")
    param = request.GET.get("param", None)
    useFR = request.GET.get("useFR", "false")
    if param is None:
        return HttpResponse("Param is illegal!", status=404)
    random_str = get_random_string(8)
    df = pd.read_csv(BASE_STATIC + username + "_corrected.csv")
    with open(BASE_STATIC + username + "_fr.json", "r") as f:
        X3D1 = json.loads(f.read())
    adata = sc.read(BASE_STATIC + username + "_adata.h5ad")
    adata_ori = adata.copy()
    if useFR == "true":
        adata1 = sc.AnnData(np.zeros(np.array(X3D1).shape), dtype=np.float64)
        adata1.X = np.array(X3D1)
        adata1.obs = adata.obs.copy()
        adata1.obs_names = adata.obs_names.copy()
        adata = adata1
    if cluster == "LEIDEN":
        if param is None:
            param = 1
        sc.pp.neighbors(adata, n_neighbors=40, n_pcs=40)
        sc.tl.leiden(adata, resolution=float(param))
        if len(set(adata.obs["leiden"])) == 1:
            # throw error for just 1 cluster
            return HttpResponse("Only 1 Cluster after clustering", status=400)

        df["cluster"] = [i for i in adata.obs["leiden"]]
        df.to_csv(BASE_STATIC + username + "_corrected_clusters.csv", index=False)

        traces = zip_for_vis(X3D1, list(adata.obs["leiden"]), adata.obs_names.tolist())

        adata_ori.obs = adata.obs.copy()
        adata_ori.obs_names = adata.obs_names.copy()
        adata_ori.write(BASE_STATIC + username + "_adata.h5ad")

        with plt.rc_context():
            sc.tl.rank_genes_groups(adata_ori, groupby="leiden", method="t-test")
            sc.tl.dendrogram(adata_ori, groupby="leiden")
            sc.pl.rank_genes_groups_dotplot(
                adata_ori, n_genes=4, show=False, color_map="bwr"
            )
            plt.savefig(
                BASE_STATIC + username + "_cluster_" + random_str + "_1.png",
                bbox_inches="tight",
            )
            sc.pl.rank_genes_groups(adata_ori, n_genes=20, sharey=False)
            plt.savefig(
                BASE_STATIC + username + "_cluster_" + random_str + "_2.png",
                bbox_inches="tight",
            )
            markers = sc.get.rank_genes_groups_df(adata_ori, None)
            markers.to_csv(BASE_STATIC + username + "_markers.csv", index=False)
        b = (
            adata.obs.sort_values(["batch1", "leiden"])
            .groupby(["batch1", "leiden"])
            .count()
            .reset_index()
        )
        # print(b)
        b = b[["batch1", "leiden", "batch2"]]
        b.columns = ["batch", "cluster", "count"]
        barChart1 = [
            {
                "x": sorted(list(set(b["cluster"].tolist()))),
                "y": b[b["batch"] == i]["count"].tolist(),
                "name": i,
                "type": "bar",
            }
            for i in set(b["batch"].tolist())
        ]
        # print(barChart1)

        b = (
            adata.obs.sort_values(["batch2", "leiden"])
            .groupby(["batch2", "leiden"])
            .count()
            .reset_index()
        )
        b = b[["batch2", "leiden", "batch1"]]
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

        return JsonResponse(
            {
                "traces": traces,
                "fileName": username + "_cluster_" + random_str + "_1.png",
                "fileName1": username + "_cluster_" + random_str + "_2.png",
                "bc1": barChart1,
                "bc2": barChart2,
            }
        )
    elif cluster == "HDBSCAN":
        if param is None:
            param = 20
        if int(param) <= 5:
            param = 5
        labels = hdbscan.HDBSCAN(min_cluster_size=int(param)).fit_predict(adata.X)
        # freq=collections.Counter(labels)
        # m=[freq[key] for key in freq]
        # m.sort()
        # if m[0]<=5:
        # 	for i in range(len(m)):
        # 		if sum(m[:(i+1)])<=5:
        # 			continue
        # 		labels=hdbscan.HDBSCAN(min_cluster_size=(m[i]+1)).fit_predict(adata.X)
        # 		break
        if len(set(labels)) == 1:
            # throw error for just 1 cluster
            return HttpResponse("Only 1 Cluster after clustering", status=400)
        labels = [str(i + 1) for i in labels]
        adata.obs["hdbscan"] = labels
        adata.obs["hdbscan"] = adata.obs["hdbscan"].astype("category")

        df["cluster"] = [i for i in adata.obs["hdbscan"]]
        df.to_csv(BASE_STATIC + username + "_corrected_clusters.csv", index=False)

        traces = zip_for_vis(X3D1, labels, adata.obs_names.tolist())

        adata_ori.obs = adata.obs.copy()
        adata_ori.obs_names = adata.obs_names.copy()
        adata_ori.write(BASE_STATIC + username + "_adata.h5ad")

        with plt.rc_context():
            sc.tl.rank_genes_groups(adata_ori, groupby="hdbscan", method="t-test")
            sc.tl.dendrogram(adata_ori, groupby="hdbscan")
            sc.pl.rank_genes_groups_dotplot(
                adata_ori, n_genes=4, show=False, color_map="bwr"
            )
            plt.savefig(
                BASE_STATIC + username + "_cluster_" + random_str + "_1.png",
                bbox_inches="tight",
            )
            sc.pl.rank_genes_groups(adata_ori, n_genes=20, sharey=False)
            plt.savefig(
                BASE_STATIC + username + "_cluster_" + random_str + "_2.png",
                bbox_inches="tight",
            )
            markers = sc.get.rank_genes_groups_df(adata_ori, None)
            markers.to_csv(BASE_STATIC + username + "_markers.csv", index=False)

        b = (
            adata.obs.sort_values(["batch1", "hdbscan"])
            .groupby(["batch1", "hdbscan"])
            .count()
            .reset_index()
        )
        b = b[["batch1", "hdbscan", "batch2"]]
        b.columns = ["batch", "cluster", "count"]
        barChart1 = [
            {
                "x": sorted(list(set(b["cluster"].tolist()))),
                "y": b[b["batch"] == i]["count"].tolist(),
                "name": i,
                "type": "bar",
            }
            for i in set(b["batch"].tolist())
        ]

        b = (
            adata.obs.sort_values(["batch2", "hdbscan"])
            .groupby(["batch2", "hdbscan"])
            .count()
            .reset_index()
        )
        b = b[["batch2", "hdbscan", "batch1"]]
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

        return JsonResponse(
            {
                "traces": traces,
                "fileName": username + "_cluster_" + random_str + "_1.png",
                "fileName1": username + "_cluster_" + random_str + "_2.png",
                "bc1": barChart1,
                "bc2": barChart2,
            }
        )
    elif cluster == "Kmeans":
        if int(param) == 1:
            return HttpResponse("Only 1 Cluster!", status=404)
        km = KMeans(n_clusters=int(param), random_state=42, n_init="auto").fit(adata.X)
        labels = [str(i) for i in km.labels_]
        adata.obs["kmeans"] = labels
        adata.obs["kmeans"] = adata.obs["kmeans"].astype("category")

        df["cluster"] = [i for i in adata.obs["kmeans"]]
        df.to_csv(BASE_STATIC + username + "_corrected_clusters.csv", index=False)

        traces = zip_for_vis(X3D1, labels, adata.obs_names.tolist())

        adata_ori.obs = adata.obs.copy()
        adata_ori.obs_names = adata.obs_names.copy()
        adata_ori.write(BASE_STATIC + username + "_adata.h5ad")

        with plt.rc_context():
            sc.tl.rank_genes_groups(adata_ori, groupby="kmeans", method="t-test")
            sc.tl.dendrogram(adata_ori, groupby="kmeans")
            sc.pl.rank_genes_groups_dotplot(
                adata_ori, n_genes=4, show=False, color_map="bwr"
            )
            plt.savefig(
                BASE_STATIC + username + "_cluster_" + random_str + "_1.png",
                bbox_inches="tight",
            )
            sc.pl.rank_genes_groups(adata_ori, n_genes=20, sharey=False)
            plt.savefig(
                BASE_STATIC + username + "_cluster_" + random_str + "_2.png",
                bbox_inches="tight",
            )
            markers = sc.get.rank_genes_groups_df(adata_ori, None)
            markers.to_csv(BASE_STATIC + username + "_markers.csv", index=False)

        b = (
            adata.obs.sort_values(["batch1", "kmeans"])
            .groupby(["batch1", "kmeans"])
            .count()
            .reset_index()
        )
        b = b[["batch1", "kmeans", "batch2"]]
        b.columns = ["batch", "cluster", "count"]
        barChart1 = [
            {
                "x": sorted(list(set(b["cluster"].tolist()))),
                "y": b[b["batch"] == i]["count"].tolist(),
                "name": i,
                "type": "bar",
            }
            for i in set(b["batch"].tolist())
        ]

        b = (
            adata.obs.sort_values(["batch2", "kmeans"])
            .groupby(["batch2", "kmeans"])
            .count()
            .reset_index()
        )
        b = b[["batch2", "kmeans", "batch1"]]
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

        return JsonResponse(
            {
                "traces": traces,
                "fileName": username + "_cluster_" + random_str + "_1.png",
                "fileName1": username + "_cluster_" + random_str + "_2.png",
                "bc1": barChart1,
                "bc2": barChart2,
            }
        )


@login_required()
def clusteringAdvanced(request):
    if request.method == "GET" and "cluster" not in request.GET:
        return render(
            request,
            "clustering_advance.html",
        )
    else:
        username = request.user.username
        cluster = request.GET.get("cluster", "LEIDEN")
        minValue = float(request.GET.get("min", "0"))
        maxValue = float(request.GET.get("max", "1"))
        level = int(request.GET.get("level", 3))
        adata = sc.read(BASE_STATIC + username + "_adata.h5ad")
        df = pd.read_csv(BASE_STATIC + username + "_corrected.csv")[["LABEL"]]
        if level > 10 or level <= 1:
            return HttpResponse("Error for the input", status=400)
        if cluster == "LEIDEN":
            if minValue <= 0:
                minValue = 0
            if maxValue >= 2:
                maxValue = 2
            for i, parami in enumerate(np.linspace(minValue, maxValue, level)):
                sc.pp.neighbors(adata, n_neighbors=40, n_pcs=40)
                sc.tl.leiden(adata, resolution=float(parami))
                df["level" + str(i + 1)] = [
                    "level" + str(i + 1) + "_" + str(j) for j in adata.obs["leiden"]
                ]
        elif cluster == "HDBSCAN":
            if minValue <= 5:
                minValue = 5
            if maxValue >= 100:
                maxValue = 100
            for i, parami in enumerate(np.linspace(minValue, maxValue, level)):
                df["level" + str(i + 1)] = [
                    "level" + str(i + 1) + "_" + str(j)
                    for j in hdbscan.HDBSCAN(min_cluster_size=int(parami)).fit_predict(
                        adata.X
                    )
                ]

        result = fromPdtoSangkey(df)
        return JsonResponse(result)




@login_required()
def advancedSearch(request):
    username = request.user.username
    name = request.GET.get("name", None)
    res = {}
    if name is None:
        return render(
            request,
            "advancedSearch.html",
        )
    name = name.replace(" ", "")
    res["name"] = name
    url = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=" + name
    page = requests.get(url)
    p = page.content.decode("utf-8").replace("\n", "")
    if "GEO accession display tool" in p:
        return HttpResponse("Could not find the cohort!", status=404)
    m = re.search(
        '(?<=\<tr valign="top"\>\<td nowrap\>Summary\</td\>\<td style="text-align: justify"\>)([\w\s,-.!"\(\):%]+)',
        p,
    )
    if m is not None:
        res["summary"] = m.group(1)
    m = re.search(
        '(?<=\<tr valign="top"\>\<td nowrap\>Overall design\</td\>\<td style="text-align: justify"\>)([\w\s,-.!"\(\):%]+)',
        p,
    )
    if m is not None:
        res["overAllDesign"] = m.group(1)
    m = re.search(
        '(\<table cellpadding="2" cellspacing="2" width="600"\>\<tr bgcolor="#eeeeee" valign="top"\>\<td align="middle" bgcolor="#CCCCCC"\>\<strong\>Supplementary file\</strong\>\</td\>)(.+)(\</tr\>\<tr\>\<td class="message"\>[\w\s]+\</td\>\</tr\>\</table\>)',
        p,
    )
    if m is not None:
        soup = BeautifulSoup(m.group(0), "html.parser")
        ftp_tags = soup.find_all("a", string="(ftp)")
        for ftp in ftp_tags:
            ftp.decompose()
        custom_tags = soup.find_all("a", string="(custom)")
        for cus in custom_tags:
            cus.decompose()
        td_tags = soup.find_all("td", class_="message")
        for td in td_tags:
            td.decompose()
        td_tags = soup.find_all("td")
        for td in td_tags:
            if "javascript:" in str(td):
                td.decompose()
        res["data"] = str(soup).replace(
            "/geo/download/", "https://www.ncbi.nlm.nih.gov/geo/download/"
        )
    if "TXT" in p:
        res["txt"] = 1
    else:
        res["txt"] = 0
    return JsonResponse(res)


@login_required()
def goenrich(request):
    username = request.user.username
    cluster_n = request.GET.get("cluster_n", 0)
    random_str = get_random_string(8)

    df = pd.read_csv(BASE_STATIC + username + "_corrected.csv")
    if (
        any(df.columns.str.startswith("c_")) is True
        or len(set(df.columns).intersection({"age", "crp", "bmi", "esr", "BMI"})) > 0
    ):
        return HttpResponse("Not Allowed Clinic Data", status=400)
    markers = pd.read_csv(BASE_STATIC + username + "_markers.csv")
    markers = markers[
        (markers.pvals_adj < 0.05)
        & (markers.logfoldchanges > 0.5)
        & (markers.group == int(cluster_n))
    ]
    if len(markers.index) == 0:
        return HttpResponse("No marker genes", status=400)
    df = go_it(markers.names.values)
    df1 = df.groupby("class").head(10).reset_index(drop=True)

    fig = go.Figure()

    def fig_add_trace_ontology(fig, df, ontology):
        fig.add_trace(
            go.Bar(
                y=df[df["class"] == ontology.name].term[::-1],
                x=df[df["class"] == ontology.name].per[::-1],
                name=ontology.name,
                customdata=[
                    "P_corr=" + str(round(i, 5)) for i in df[df["class"] == ontology.name].p_corr[::-1]
                ],
                hovertemplate="Ratio: %{x:.5f}<br> %{customdata}",
                orientation="h",
                marker={
                    "color": df[df["class"] == ontology.name].p_corr[::-1],
                    "colorscale": ontology.color,
                }
            )
        )
    for o in ONTOLOGY.values():
        fig_add_trace_ontology(fig, df1, o)

    fig.update_layout(
        barmode="stack",
        height=1200,
        uniformtext_minsize=50,
        uniformtext_mode="hide",
        xaxis=dict(title="Gene Ratio"),
    )

    fig.write_image(BASE_STATIC + username + "_goenrich_" + random_str + ".png")
    # return render(request,'goenrich.html',{'fileName':username+'_goenrich_'+random_str+'.png'})
    return JsonResponse({"fileName": username + "_goenrich_" + random_str + ".png"})


@login_required()
def lasso(request):
    username = request.user.username
    random_str = get_random_string(8)
    cluster = int(request.GET.get("cluster_n", 0))  # +1 for R
    # stdout_file=BASE_STATIC+username+'_'+random_str+'_stdout.txt'
    # stderr_file=BASE_STATIC+username+'_'+random_str+'_stderr.txt'
    # code=400
    # print('Rscript lasso.R '+username+' '+str(cluster))
    # try:
    # 	with open(stdout_file,'w')as out, open(stderr_file,'w')as err:
    # 		Popen(['Rscript --max-ppsize=5000000 lasso.R '+username+' '+str(cluster)+' '+random_str],stdout=out,stderr=err,shell=True).communicate()#calling Rscript
    # 	with open(stdout_file,'r')as out, open(stderr_file,'r')as err:
    # 		stdout=out.read()
    # 		stderr=err.read()
    # 	code=200
    # except:
    # 	retValueOut=str(stderr)
    # 	code=400
    # finally:
    # 	if code==200:
    # 		if os.path.exists(stdout_file):
    # 			os.remove(stdout_file)
    # 		if os.path.exists(stderr_file):
    # 			os.remove(stderr_file)
    # 	else:
    # 		return HttpResponse('Error happened on the Server',status=code)

    df = pd.read_csv(BASE_STATIC + username + "_corrected_clusters.csv")
    df.drop(["FileName", "LABEL", "obs"], axis=1, inplace=True)
    x = df.drop(["cluster"], axis=1, inplace=False)
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)
    index = df["cluster"] == cluster
    index1 = df["cluster"] != cluster
    df.loc[index, "cluster"] = 1
    df.loc[index1, "cluster"] = 0
    y = pd.Categorical(df.cluster)
    model = LassoCV(cv=5, max_iter=10000, random_state=42)
    model.fit(x, y)

    lasso_tuned = Lasso().set_params(alpha=model.alpha_)
    lasso_tuned.fit(x, y)
    coef = pd.Series(
        lasso_tuned.coef_, df.drop(["cluster"], axis=1, inplace=False).columns
    ).sort_values(key=abs, ascending=False)
    
    matplotlib.pyplot.clf()  # in order to save a picture
    coef[coef != 0][:50].plot.bar(x="Features", y="Coef")  
    plt.savefig(
        BASE_STATIC + username + "_" + random_str + "_lasso.png", bbox_inches="tight"
    )
    return JsonResponse({"fileName": username + "_" + random_str + "_lasso.png"})



