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
from threadpoolctl import threadpool_limits

from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import math
import io
import base64

# gene_result.txt, genes_ncbi_proteincoding.py, go-basic.obo

import matplotlib
import re

matplotlib.use("agg")
import plotly.graph_objects as go

import requests
from bs4 import BeautifulSoup
from .constants import ONTOLOGY, BUILT_IN_LABELS
from .utils import (
    zip_for_vis,
    fromPdtoSangkey,
    go_it,
    combat,
    harmony,
    bbknn,
    clusteringPostProcess,
    getTopGeneCSV,
    vlnPlot,
    densiPlot,
    heatmapPlot,
    GeneID2SymID,
    usrCheck,
    UploadFileColumnCheck,
)

from .models import userData, MetaFileColumn, CustomUser, UploadedFile, SharedFile
from django.db import transaction


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
    context = {}
    if request.user.groups.exists():
        context["cohorts"] = list(
            SharedFile.objects.filter(
                type1="expression", groups__in=request.user.groups.all()
            )
            .all()
            .values_list("cohort", "label")
        )
    else:
        context["cohorts"] = list(
            SharedFile.objects.filter(type1="expression")
            .all()
            .values_list("cohort", "label")
        )
    return render(request, "tab1.html", {"root": context})


@login_required()
def uploadExpression(request):
    if request.method != "POST":
        return HttpResponse("Method not allowed", status=405)
    username = request.user.username
    cID = request.POST.get("cID", None)
    if cID is None:
        return HttpResponse("cID not provided.", status=400)
    fileNames = []
    context = {}
    UploadedFile.objects.filter(user=request.user, type1="exp", cID=cID).delete()
    files = request.FILES.getlist("files[]", None)
    new_exp_files = []

    for f in files:
        temp_file = UploadedFile(user=request.user, cID=cID, type1="exp", file=f)
        new_exp_files.append(temp_file)
        fileNames.append(temp_file.file.path)
        context[f.name] = {}
    UploadedFile.objects.bulk_create(new_exp_files)

    for file in UploadedFile.objects.filter(
        user=request.user, type1="exp", cID=cID
    ).all():
        df = pd.read_csv(file.file.path, nrows=5, header=0)
        f = file.file.name
        if "ID_REF" not in df.columns:
            return HttpResponse("No ID_REF column in the expression file", status=400)
        if UploadFileColumnCheck(df) == 0:
            return HttpResponse("Column name illegal.", status=400)
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
    cID = request.POST.get("cID", None)
    if cID is None:
        return HttpResponse("cID not provided.", status=400)
    files = request.FILES.getlist("meta", None)
    if files is None:
        return HttpResponse("Upload the meta file is required", status=405)
    files = files[0]
    UploadedFile.objects.filter(user=request.user, type1="cli", cID=cID).delete()
    UploadedFile.objects.create(user=request.user, cID=cID, type1="cli", file=files)
    f = UploadedFile.objects.filter(user=request.user, type1="cli", cID=cID).first()
    context = {}
    context["metaFile"] = {}
    df = pd.read_csv(f.file.path, nrows=5, header=0)
    if "ID_REF" not in df.columns:
        return HttpResponse("No ID_REF column in the clinical file", status=400)
    if "LABEL" not in df.columns:
        return HttpResponse("No LABEL column in the clinical file", status=400)
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
def edaIntegrate(request):
    checkRes = usrCheck(request, 0)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]
    username = request.user.username
    cID = request.GET.get("cID", None)

    corrected = request.GET.get("correct", "Combat")
    log2 = request.GET.get("log2", "No")
    fr = request.GET.get("fr", "TSNE")
    integrate = [i.strip() for i in request.GET.get("integrate", "").split(",")]
    if "" in integrate:
        integrate.remove("")

    files = UploadedFile.objects.filter(user=request.user, type1="exp", cID=cID).all()
    if files:
        files = [i.file.path for i in files]
    else:
        files = []

    files_meta = set()
    in_ta, in_ta1 = {}, {}
    if request.user.groups.exists():
        for i in SharedFile.objects.filter(
            type1="expression", groups__in=request.user.groups.all()
        ).all():
            in_ta[i.cohort] = i.file.path
        for i in SharedFile.objects.filter(
            type1="meta", groups__in=request.user.groups.all()
        ).all():
            in_ta1[i.cohort] = i.file.path
    else:
        for i in SharedFile.objects.filter(type1="expression").all():
            in_ta[i.cohort] = i.file.path
        for i in SharedFile.objects.filter(type1="meta").all():
            in_ta1[i.cohort] = i.file.path

    for i in integrate:
        if i in in_ta:
            files.append(in_ta[i])
            files_meta.add(in_ta1[i])
    dfs = []
    batch = []
    obs = []
    temp0 = []

    f = UploadedFile.objects.filter(user=request.user, type1="cli", cID=cID).first()
    if f is not None:
        temp0 = pd.read_csv(f.file.path)
        temp0 = temp0.dropna(axis=1)
    else:
        temp0 = pd.DataFrame()
    if integrate[0] != "null" or integrate != [""]:  # jquery plugin compatible
        for i in files_meta:
            if temp0.shape == (0, 0):
                temp0 = pd.concat(
                    [temp0, pd.read_csv(i).dropna(axis=1, inplace=False)],
                    axis=0,
                    join="outer",
                )
            else:
                temp0 = pd.concat(
                    [temp0, pd.read_csv(i).dropna(axis=1, inplace=False)],
                    axis=0,
                    join="inner",
                )
    if temp0.shape == (0, 0):
        return HttpResponse("Can't find meta file", status=400)
    if len(files) == 0:
        return HttpResponse("Can't find expression file", status=400)

    for file in files:
        temp01 = pd.read_csv(file).set_index("ID_REF")
        if "raw" in file or "RAW" in file:
            temp01 = temp01.div(temp01.sum(axis=1), axis=0) * 1e6
            # temp01=temp01/temp01.sum()*1e6
        temp = temp01.join(temp0[["ID_REF", "LABEL"]].set_index("ID_REF"), how="inner")
        temp1 = temp.reset_index().drop(
            ["ID_REF", "LABEL"], axis=1, inplace=False
        )  # exclude ID_REF & LABEL
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
    elif len(dfs) == 0:
        return HttpResponse("No matched data for meta and omics", status=400)
    else:
        dfs1 = dfs[0]

    dfs1["ID_REF"] = obs
    dfs1["FileName"] = batch
    # temp0=pd.concat(temp0,axis=0).reset_index(drop=True) #combine all clinic data
    temp = dfs1.set_index("ID_REF").join(temp0.set_index("ID_REF"), how="inner")
    temp["obs"] = temp.index.tolist()
    usr.setIntegrationData(temp)

    pca_temp = usr.getAnndata().obsm["X_pca"]

    if fr == "TSNE":
        tsne = TSNE(n_components=2, random_state=42, n_jobs=2)
        X2D = tsne.fit_transform(pca_temp)
    else:
        umap1 = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, n_jobs=2)
        X2D = umap1.fit_transform(pca_temp)

    usr.setFRData(X2D)

    if usr.save() is False:
        return HttpResponse("Error for creating user records.", status=400)

    # store column info of meta file into database
    try:
        with transaction.atomic():
            new_file_columns = []
            MetaFileColumn.objects.filter(user=request.user, cID=cID).delete()
            for cn in temp0.columns:
                if cn == "LABEL":
                    label = "1"
                else:
                    label = "0"
                temp_meta = MetaFileColumn(
                    user=request.user, cID=cID, colName=cn, label=label
                )
                if temp_meta is None:
                    raise Exception("MetaFileColumn create Failed.")
                else:
                    new_file_columns.append(temp_meta)
            MetaFileColumn.objects.bulk_create(new_file_columns)
    except:
        return HttpResponse("Error for registering to DataBase.", status=400)

    return HttpResponse("Operation successful.", status=200)


@login_required()
def eda(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]

    adata = usr.getAnndata()
    targetLabel = request.GET.get("label", "batch2")
    color2 = adata.obs[targetLabel]  # temp.LABEL, temp.FileName
    X2D = usr.getFRData()
    traces = zip_for_vis(
        X2D.tolist(), adata.obs["batch1"], adata.obs_names
    )  # temp.FileName, temp.obs
    traces1 = zip_for_vis(X2D.tolist(), color2, adata.obs_names)
    labels = list(
        MetaFileColumn.objects.filter(user=request.user, cID=usr.cID, label="1")
        .all()
        .values_list("colName", flat=True)
    )
    if "LABEL" in labels:
        labels.remove("LABEL")
        labels = ["LABEL"] + labels
    context = {
        "dfs1": traces,
        "dfs2": traces1,
        "labels": labels,
    }
    return JsonResponse(context)


@login_required()
def dgea(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]
    targetLabel = request.GET.get("label", "batch2")
    adata = usr.getAnndata()

    clusters = request.GET.get("clusters", "default")
    n_genes = request.GET.get("topN", 4)

    if clusters == "default":
        with plt.rc_context():
            figure1 = io.BytesIO()
            figure2 = io.BytesIO()
            if len(set(adata.obs["batch1"])) > 1:
                sc.tl.rank_genes_groups(adata, groupby="batch1", method="t-test")
                sc.tl.dendrogram(adata, groupby="batch1")
                sc.pl.rank_genes_groups_dotplot(
                    adata, n_genes=int(n_genes), show=False, color_map="bwr"
                )
                plt.savefig(
                    figure1,
                    format="png",
                    bbox_inches="tight",
                )
                figure1 = base64.b64encode(figure1.getvalue()).decode("utf-8")
            else:
                figure1 = ""
            if len(set(adata.obs[targetLabel])) > 1:
                sc.tl.rank_genes_groups(adata, groupby=targetLabel, method="t-test")
                sc.tl.dendrogram(adata, groupby=targetLabel)
                sc.pl.rank_genes_groups_dotplot(
                    adata, n_genes=int(n_genes), show=False, color_map="bwr"
                )
                plt.savefig(
                    figure2,
                    format="png",
                    bbox_inches="tight",
                )
                figure2 = base64.b64encode(figure2.getvalue()).decode("utf-8")
            else:
                figure2 = ""
            return JsonResponse([figure1, figure2], safe=False)
    elif clusters == "fileName":
        # show top gene for specific group
        return getTopGeneCSV(adata, "batch1", n_genes)
    elif clusters == "label":
        return getTopGeneCSV(adata, targetLabel, n_genes)
    elif clusters in ("LEIDEN", "HDBSCAN", "KMeans"):
        return getTopGeneCSV(adata, "cluster", n_genes)


@login_required()
def clustering(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]
    username = request.user.username
    clientID = request.GET.get("cID", None)

    cluster = request.GET.get("cluster", "LEIDEN")
    param = request.GET.get("param", None)
    if param is None:
        return HttpResponse("Param is illegal!", status=400)
    X2D = usr.getFRData()
    if X2D is None:
        return HttpResponse("Please run feature reduction first.", status=400)
    X2D = X2D.tolist()
    adata = usr.getAnndata()
    if cluster == "LEIDEN":
        if param is None:
            param = 1
        try:
            param = float(param)
        except:
            return HttpResponse("Resolution should be a float", status=400)
        sc.pp.neighbors(adata, n_neighbors=40, n_pcs=40)
        sc.tl.leiden(adata, resolution=param)
        Resp = clusteringPostProcess(X2D, adata, "leiden", usr)
        return Resp
    elif cluster == "HDBSCAN":
        if param is None:
            param = 20
        try:
            param = int(param)
        except:
            return HttpResponse("K should be positive integer.", status=400)
        if param <= 5:
            param = HttpResponse("minSize should be at least 5.", status=400)
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
            return HttpResponse("K should be positive integer.", status=400)

        if param <= 1:
            return HttpResponse("K should be larger than 1.", status=400)
        km = KMeans(n_clusters=int(param), random_state=42, n_init="auto").fit(
            adata.obsm["X_pca"]
        )
        labels = [str(i) for i in km.labels_]
        adata.obs["kmeans"] = labels
        adata.obs["kmeans"] = adata.obs["kmeans"].astype("category")
        Resp = clusteringPostProcess(X2D, adata, "kmeans", usr)
        return Resp


@login_required()
def clusteringAdvanced(request):
    clientID = request.GET.get("cID", None)
    if clientID is None:
        return HttpResponse("clientID is Required", status=400)
    if request.method == "GET" and "cluster" not in request.GET:
        return render(request, "clustering_advance.html", {"cID": clientID})
    else:
        checkRes = usrCheck(request)
        if checkRes["status"] == 0:
            return HttpResponse(checkRes["message"], status=400)
        else:
            usr = checkRes["usrData"]
        cluster = request.GET.get("cluster", "LEIDEN")
        minValue = float(request.GET.get("min", "0"))
        maxValue = float(request.GET.get("max", "1"))
        level = int(request.GET.get("level", 3))
        adata = usr.getAnndata().copy()
        df = usr.getCorrectedCSV()[["LABEL"]]
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
                        adata.obsm["X_pca"]
                    )
                ]

        result = fromPdtoSangkey(df)
        return JsonResponse(result)


@login_required()
def advancedSearch(request):
    # username = request.user.username
    name = request.GET.get("name", None)
    clientID = request.GET.get("cID", None)
    if clientID is None:
        return HttpResponse("clientID is Required", status=400)
    res = {}
    if name is None:
        return render(request, "advancedSearch.html", {"clientID": clientID})
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
            ftp.decompose()  # remove all ftp nodes
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
    cluster_n = request.GET.get("cluster_n", 0)
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]
    df = usr.getCorrectedCSV()
    if (
        any(df.columns.str.startswith("c_")) is True
        or len(set(df.columns).intersection({"age", "crp", "bmi", "esr", "BMI"})) > 0
    ):
        return HttpResponse("Not Allowed Clinic Data", status=400)
    markers = usr.getMarkers()
    if markers is None:
        return HttpResponse("Please run clustering method first.", status=400)
    markers = markers[
        (markers.pvals_adj < 0.05)
        & (markers.logfoldchanges > 0.5)
        & (markers.group.astype(int) == int(cluster_n))
    ]
    if len(markers.index) == 0:
        return HttpResponse("No marker genes", status=400)
    with threadpool_limits(limits=2, user_api="blas"):
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
                    "P_corr=" + str(round(i, 5))
                    for i in df[df["class"] == ontology.name].p_corr[::-1]
                ],
                hovertemplate="Ratio: %{x:.5f}<br> %{customdata}",
                orientation="h",
                marker={
                    "color": df[df["class"] == ontology.name].p_corr[::-1],
                    "colorscale": ontology.color,
                },
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
    return HttpResponse(
        base64.b64encode(fig.to_image(format="png")), content_type="image/png"
    )


@login_required()
def lasso(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]

    cluster = int(request.GET.get("cluster_n", 0))  # +1 for R
    adata = usr.getAnndata()
    df = adata.to_df().round(12)
    if "cluster" not in adata.obs.columns:
        return HttpResponse("Please run clustering method first.", status=400)
    df["cluster"] = adata.obs["cluster"].astype(int)
    x = df.drop(["cluster"], axis=1, inplace=False)

    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)

    index = df["cluster"] == cluster
    index1 = df["cluster"] != cluster
    df.loc[index, "cluster"] = 1
    df.loc[index1, "cluster"] = 0
    y = pd.Categorical(df.cluster)
    model = LassoCV(cv=5, random_state=42, n_jobs=-1, max_iter=10000, tol=0.01)
    model.fit(x, y)

    coef = pd.Series(
        model.coef_, df.drop(["cluster"], axis=1, inplace=False).columns
    ).sort_values(key=abs, ascending=False)

    coef[coef != 0][:50].plot.bar(
        x="Features", y="Coef", figure=plt.figure(), fontsize=8
    )
    with io.BytesIO() as buffer:
        plt.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        image = buffer.getvalue()
    return HttpResponse(base64.b64encode(image), content_type="image/png")


@login_required()
def downloadCorrected(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]
    result_df = usr.getCorrectedCSV()
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = "attachment; filename=corrected.csv"
    result_df.to_csv(path_or_buf=response)
    return response


@login_required()
def downloadCorrectedCluster(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]

    result_df = usr.getCorrectedClusterCSV()
    if result_df is None:
        return HttpResponse("Please do clustering first.", status=400)
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = "attachment; filename=correctedCluster.csv"
    result_df.to_csv(path_or_buf=response)
    return response


@login_required()
def candiGenes(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]

    method = request.GET.get("method", None)
    maxGene = 12
    if method is None or method == "pca":
        adata = usr.getAnndata()
        n_pcs = 3
        pcs_loadings = pd.DataFrame(adata.varm["PCs"][:, :n_pcs], index=adata.var_names)
        pcs_loadings.dropna(inplace=True)
        result = []
        for i in pcs_loadings.columns:
            result.extend(pcs_loadings.nlargest(2, columns=i).index.tolist())
            result.extend(pcs_loadings.nsmallest(2, columns=i).index.tolist())
        return JsonResponse(result, safe=False)
    else:
        markers = usr.getMarkers()
        if markers is None:
            return HttpResponse("Please run clustering method first.", status=400)
        clusters = set(markers.group)
        number = math.ceil(maxGene / len(clusters))
        result = (
            markers.groupby("group")
            .apply(lambda x: x.nlargest(number, "scores"))
            .names.tolist()
        )
        return JsonResponse(result, safe=False)


@login_required()
def genePlot(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]
    type = request.GET.get("type", "vln")
    geneList = request.GET.get("geneList", None)
    if geneList is None:
        return HttpResponse("geneList is Required", status=400)
    try:
        geneList = geneList.split(",")
        geneList = [i for i in geneList if i != "None"]
    except:
        return HttpResponse("geneList is illigal", status=400)

    adata = usr.getAnndata()
    if adata is None:
        return HttpResponse("No data is in use for the account", status=400)
    geneList1 = []
    for i in geneList:  # find legal geneList
        if i in adata.var_names:
            geneList1.append(i)
    if len(geneList1) == 0:
        return HttpResponse("Can't find legal gene in the list", status=400)
    X2D = usr.getFRData()
    if X2D is None:
        return HttpResponse("Please run feature reduction first.", status=400)
    if len(geneList1) > 12:
        geneList1 = geneList1[:12]
    if type == "vln":
        return vlnPlot(geneList1, adata)
    elif type == "density":
        return densiPlot(geneList1, adata)
    else:  # type=='heatmap'
        return heatmapPlot(geneList1, adata)


@login_required()
def GeneLookup(request):
    geneList = request.GET.get("geneList", None)
    if geneList is None:
        return HttpResponse("geneList is Required", status=400)
    try:
        geneList = geneList.split(",")
        geneList = [i for i in geneList if i != "None"]
    except:
        return HttpResponse("geneList is illigal", status=400)
    result = GeneID2SymID(geneList)
    if result is None:
        return HttpResponse("geneList is illigal", status=400)
    return JsonResponse(result, safe=False)


@login_required()
def checkUser(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        return HttpResponse("User exists.", status=200)


@login_required()
def meta_columns(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]
    cID = request.GET.get("cID", None)
    if request.method == "GET":
        result = [
            [i[0], i[1]]
            for i in MetaFileColumn.objects.filter(user=request.user, cID=usr.cID)
            .all()
            .values_list("colName", "label")
        ]
        return JsonResponse(result, safe=False)
    if request.method == "PUT":
        labels = request.GET.get("labels", None)
        fr = request.GET.get("fr", "TSNE")
        if labels is None:
            return HttpResponse("Labels illegal.", status=400)
        labels = [i.strip() for i in labels.split(",")]
        error_labels = BUILT_IN_LABELS.intersection(set(labels))
        if len(error_labels) != 0:
            return HttpResponse(
                "Labels creating Problem, can't use retained words as an label:"
                + str(error_labels),
                status=400,
            )
        try:
            with transaction.atomic():
                df = usr.getIntegrationData().copy()
                if labels != [""]:
                    df1 = df.drop(labels, axis=1, inplace=False)
                    usr.setAnndata(df1)
                    adata = usr.getAnndata()
                    for label in labels:
                        if not np.issubdtype(df[label].dtype, np.number):
                            adata.obs[label] = df[label]
                        else:
                            raise Exception(
                                "Can't auto convert numerical value for label."
                            )
                    MetaFileColumn.objects.exclude(colName="LABEL").exclude(
                        user=request.user, cID=cID, colName__in=labels
                    ).update(label="0")
                    MetaFileColumn.objects.filter(
                        user=request.user, cID=cID, colName__in=labels
                    ).update(label="1")
                else:
                    usr.setIntegrationData(df)
                    MetaFileColumn.objects.exclude(colName="LABEL").filter(
                        user=request.user, cID=cID
                    ).update(label="0")

                pca_temp = usr.getAnndata().obsm["X_pca"]
                if fr == "TSNE":
                    tsne = TSNE(n_components=2, random_state=42, n_jobs=2)
                    X2D = tsne.fit_transform(pca_temp)
                else:
                    umap1 = umap.UMAP(
                        n_components=2, random_state=42, n_neighbors=30, n_jobs=2
                    )
                    X2D = umap1.fit_transform(pca_temp)
                usr.setFRData(X2D)
        except Exception as e:
            return HttpResponse("Labels creating Problem. " + str(e), status=400)
        finally:
            usr.save()
        return HttpResponse("Labels updated successfully.", status=200)
