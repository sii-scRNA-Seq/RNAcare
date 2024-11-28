from django.shortcuts import render
from django.http import HttpResponse, JsonResponse

# from django.contrib.auth.decorators import login_required
import pandas as pd
import json
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap.umap_ as umap
import scanpy as sc
import numpy as np
from matplotlib import pyplot as plt
import hdbscan
from threadpoolctl import threadpool_limits
from sklearn.preprocessing import StandardScaler
import math
import io
import base64
from django.db.models import Q
from collections import defaultdict
from django.contrib.auth import authenticate

# gene_result.txt, genes_ncbi_proteincoding.py, go-basic.obo

import matplotlib
import re

matplotlib.use("agg")
import plotly.graph_objects as go

import requests
from bs4 import BeautifulSoup
from .constants import BUILT_IN_LABELS, NUMBER_CPU_LIMITS, ALLOW_UPLOAD
from .utils import (
    zip_for_vis,
    fromPdtoSangkey,
    GeneID2SymID,
    usrCheck,
    getExpression,
    getMeta,
    generate_jwt_token,
    auth_required,
)

from .models import MetaFileColumn, UploadedFile, SharedFile
from django.db import transaction
from IMID.tasks import (
    vlnPlot,
    densiPlot,
    heatmapPlot,
    runLasso,
    runIntegrate,
    runDgea,
    runClustering,
    runGoEnrich,
    runFeRed,
)

ALLOW_UPLOAD = True
# from pydeseq2.ds import DeseqStats


def restLogin(request):
    if request.method != "POST":
        return HttpResponse("Method not allowed.", status=405)
    body = request.body.decode("utf-8")
    try:
        data = json.loads(body)
    except:
        return JsonResponse({"error": "Invalid credentials"}, status=400)
    username = data["username"]
    password = data["password"]
    user = authenticate(username=username, password=password)
    if user is not None:
        token = generate_jwt_token(user)
        return JsonResponse({"token": token})
    return JsonResponse({"error": "Invalid credentials"}, status=400)


@auth_required
def index(request):
    return render(
        request,
        "index.html",
    )


"""
This page will show different shared data to different users within different user groups. If the user is defined in a group,
the programme will filter his corresponding group. Otherwise it will return all shared data.
"""


@auth_required
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


"""
This program provides entrance for user to upload/get Expression files. One user can upload multiple expression files at a time.
'ID_REF' should be one of the colnames in each Expression file.
User uses UploadedFile to store their uploaded expression data, the column type1 is set to be 'exp' to represent the stored expression file.
"""


@auth_required
def opExpression(request):
    if request.method == "POST":
        if ALLOW_UPLOAD is False:
            return HttpResponse("Not Allowed user upload", status=400)
        cID = request.POST.get("cID", None)
        if cID is None:
            return HttpResponse("cID not provided.", status=400)
        UploadedFile.objects.filter(user=request.user, type1="exp", cID=cID).delete()
        files = request.FILES.getlist("files[]", None)
        new_exp_files = []

        for f in files:
            temp_file = UploadedFile(user=request.user, cID=cID, type1="exp", file=f)
            new_exp_files.append(temp_file)
        UploadedFile.objects.bulk_create(new_exp_files)
        return getExpression(request, cID)
    elif request.method == "GET":
        cID = request.GET.get("cID", None)
        if cID is None:
            return HttpResponse("cID not provided.", status=400)
        return getExpression(request, cID, 0)


"""
This program provides entrance for user to upload/get Meta files. One user can upload only one meta file at a time.
'ID_REF','LABEL' should be one of the colnames in each Meta file.
User uses UploadedFile to store their uploaded expression data, the column type1 is set to be 'cli' to represent the stored clinical data.
"""


@auth_required
def opMeta(request):
    if request.method == "POST":
        if ALLOW_UPLOAD is False:
            return HttpResponse("Not Allowed user upload", status=400)
        cID = request.POST.get("cID", None)
        if cID is None:
            return HttpResponse("cID not provided.", status=400)
        files = request.FILES.getlist("meta", None)
        if files is None:
            return HttpResponse("Upload the meta file is required", status=405)
        files = files[0]
        UploadedFile.objects.filter(user=request.user, type1="cli", cID=cID).delete()
        UploadedFile.objects.create(user=request.user, cID=cID, type1="cli", file=files)
        return getMeta(request, cID)
    elif request.method == "GET":
        cID = request.GET.get("cID", None)
        if cID is None:
            return HttpResponse("cID not provided.", status=400)
        return getMeta(request, cID, 0)


"""
This is for data integration based on selected options. The files comes from 1. user uploaded files. 2. Built-in Data.
in_ta and in_ta1 are used for expression and meta data list separately
The processing logic is:
First, using user uploaded meta file as the base then inner join with selected built-in meta files to get the shared clinic features. =>temp0
Then join the expression files=>dfs1 with consideration of log2, batch effect
Third, join temp0 and dfs1;
"""


@auth_required
def edaIntegrate(request):
    checkRes = usrCheck(request, 0)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]
    cID = request.GET.get("cID", None)

    corrected = request.GET.get("correct", "Combat")
    log2 = request.GET.get("log2", "No")
    fr = request.GET.get("fr", "TSNE")
    integrate = [i.strip() for i in request.GET.get("integrate", "").split(",")]
    if "" in integrate:
        integrate.remove("")

    try:
        result = runIntegrate.apply_async(
            (request, integrate, cID, log2, corrected, usr, fr), serializer="pickle"
        ).get()
    except Exception as e:
        return HttpResponse(str(e), status=500)
    return HttpResponse("Operation successful.", status=200)


@auth_required
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
        "method": usr.redMethod,
    }
    return JsonResponse(context)


@auth_required
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

    try:
        result = runDgea.apply_async(
            (clusters, adata, targetLabel, n_genes), serializer="pickle"
        ).get()
    except Exception as e:
        return HttpResponse(str(e), status=500)
    if type(result) is list:
        return JsonResponse(result, safe=False)
    else:
        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = "attachment; filename=topGenes.csv"
        result.to_csv(path_or_buf=response)
        return response


@auth_required
def clustering(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]

    cluster = request.GET.get("cluster", "LEIDEN")
    param = request.GET.get("param", None)
    if param is None:
        return HttpResponse("Param is illegal!", status=400)
    X2D = usr.getFRData()
    if X2D is None:
        return HttpResponse("Please run feature reduction first.", status=400)
    X2D = X2D.tolist()
    adata = usr.getAnndata()
    try:
        result = runClustering.apply_async(
            (cluster, adata, X2D, usr, param), serializer="pickle"
        ).get()
    except Exception as e:
        return HttpResponse(str(e), status=400)
    return JsonResponse(result)


@auth_required
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
                with threadpool_limits(limits=NUMBER_CPU_LIMITS, user_api="blas"):
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


@auth_required
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


@auth_required
def goenrich(request):
    cluster_n = request.GET.get("cluster_n", None)
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]

    colName = request.GET.get("colName", None)
    if cluster_n is None or colName is None:
        return HttpResponse("Illegal colName/value for the Label.", status=400)

    try:
        result = runGoEnrich.apply_async(
            (usr, colName, cluster_n), serializer="pickle"
        ).get()
    except Exception as e:
        return HttpResponse(str(e), status=400)
    return HttpResponse(result, content_type="image/svg+xml")


@auth_required
def lasso(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]

    cluster = request.GET.get("cluster_n", None)  # +1 for R
    colName = request.GET.get("colName", None)
    if cluster is None or colName is None:
        return HttpResponse("Illegal colName/value for the Label.", status=400)
    adata = usr.getAnndata()
    if colName not in adata.obs.columns:
        return HttpResponse("Illegal colName for the Label.", status=400)
    df = adata.to_df().round(12)
    df[colName] = adata.obs[colName].astype(str)
    x = df.drop([colName], axis=1, inplace=False)

    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)

    index = df[colName] == cluster
    index1 = df[colName] != cluster
    df.loc[index, colName] = "1"
    df.loc[index1, colName] = "0"
    y = pd.Categorical(df[colName])

    try:
        image = runLasso.apply_async((x, y, df, colName), serializer="pickle").get()
        if image == b"":
            return HttpResponse("No features after filtering.", status=400)
    except Exception as e:
        return HttpResponse("Lasso Failed:" + str(e), status=500)
    return HttpResponse(image, content_type="image/svg+xml")


@auth_required
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


@auth_required
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


@auth_required
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
        markers = usr.getMarkers(method)
        if markers is None:
            return HttpResponse("Please run clustering method first.", status=400)
        try:
            clusters = set(markers.group)
        except Exception as e:
            return HttpResponse(
                "There is no different values for the label you chose.", status=400
            )
        number = math.ceil(maxGene / len(clusters))
        result = (
            markers.groupby("group")
            .apply(lambda x: x.nlargest(number, "scores"))
            .names.tolist()
        )
        return JsonResponse(result, safe=False)


@auth_required
def genePlot(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]
    type = request.GET.get("type", "vln")
    geneList = request.GET.get("geneList", None)
    groupby = request.GET.get("groupby", "cluster")
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
    if (
        "cluster" not in adata.obs.columns
        and groupby == "cluster"
        and type != "density"
    ):
        return HttpResponse("Please run clustering first.", status=400)
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
        image_data = vlnPlot.apply_async(
            (geneList1, adata, groupby), serializer="pickle"
        )
    elif type == "density":
        image_data = densiPlot.apply_async((geneList1, adata), serializer="pickle")
    else:  # type=='heatmap'
        image_data = heatmapPlot.apply_async(
            (geneList1, adata, groupby), serializer="pickle"
        )
    return JsonResponse({"fileName": image_data.get()})


@auth_required
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


@auth_required
def checkUser(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        return HttpResponse("User exists.", status=200)


@auth_required
def meta_columns(request):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]
    cID = request.GET.get("cID", None)
    if request.method == "GET":
        numeric = request.GET.get("numeric", None)
        if numeric is None:
            result = [
                [i[0], i[1], i[2]]
                for i in MetaFileColumn.objects.filter(user=request.user, cID=usr.cID)
                .all()
                .values_list("colName", "label", "numeric")
            ]
        else:
            result = [
                [i[0], i[1], i[2]]
                for i in MetaFileColumn.objects.filter(
                    user=request.user, cID=usr.cID, numeric=numeric
                )
                .all()
                .values_list("colName", "label", "numeric")
            ]
        return JsonResponse(result, safe=False)
    elif request.method == "PUT":
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
                crtedDic = defaultdict(list)
                for i in df.columns:
                    if "__crted" in i:
                        crtedDic[i.split("__crted")[0] + "__crted"].append(i)
                if labels != [""]:
                    labels_t = labels.copy()
                    labels1 = set(
                        [
                            i.split("__crted")[0] + "__crted"
                            for i in labels
                            if "__crted" in i
                        ]
                    )
                    labels2 = [i for i in labels if "__crted" not in i]
                    labels = set()
                    for i in crtedDic:
                        if i in labels1:
                            labels.update(crtedDic[i])  # add age__crted1-N
                            labels.add(i.split("__crted")[0])  # add age
                        else:
                            labels.update(crtedDic[i])  # add agg_crted1-N
                    labels.update(labels2)
                    df1 = df.drop(labels, axis=1, inplace=False)
                    usr.setAnndata(df1)
                    adata = usr.getAnndata()
                    for label in labels_t:
                        if not np.issubdtype(df[label].dtype, np.number):
                            adata.obs[label] = df[label]
                        else:
                            raise Exception(
                                "Can't auto convert numerical value for label."
                            )

                    MetaFileColumn.objects.exclude(colName="LABEL").exclude(
                        user=request.user, cID=cID, colName__in=labels_t
                    ).update(label="0")
                    MetaFileColumn.objects.filter(
                        user=request.user, cID=cID, colName__in=labels_t
                    ).update(label="1")
                else:
                    usr.setIntegrationData(df)
                    MetaFileColumn.objects.exclude(colName="LABEL").filter(
                        user=request.user, cID=cID
                    ).update(label="0")
                X2D = runFeRed.apply_async((fr, usr), serializer="pickle").get()
                usr.setFRData(X2D)
        except Exception as e:
            return HttpResponse("Labels creating Problem. " + str(e), status=400)
        finally:
            usr.save()
        return HttpResponse("Labels updated successfully.", status=200)
    elif request.method == "POST":
        post_data = json.loads(request.body)
        colName = post_data.get("colName")
        threshold = post_data.get("thredshold")
        threshold_labels = post_data.get("thredshold_labels")
        if (
            colName is None
            or threshold is None
            or threshold_labels is None
            or len(threshold) != len(threshold_labels) - 1
        ):
            return HttpResponse("Illegal Param Input. ", status=400)

        threshold = [float(i) for i in threshold]
        count = MetaFileColumn.objects.filter(
            (Q(colName=colName) | Q(colName__startswith=colName + "__crted"))
            & Q(user=request.user)
        ).count()
        if count == 0:
            HttpResponse(
                "No such colName: " + str(colName) + " in meta file.", status=400
            )
        colName1 = colName + "__crted" + str(count)
        df, adata = usr.getIntegrationData(), usr.getAnndata()
        conditions = [df[colName] <= threshold[0]]
        for i in range(len(threshold_labels) - 2):
            conditions.append(
                (df[colName] > threshold[i]) & (df[colName] <= threshold[i + 1])
            )
        conditions.append(df[colName] >= threshold[-1])

        try:
            with transaction.atomic():
                df[colName1] = np.select(conditions, threshold_labels)
                adata.obs[colName1] = df[colName1].copy()
                MetaFileColumn.objects.create(
                    user=request.user, cID=cID, colName=colName1, label="0", numeric="0"
                )
        except Exception as e:
            return HttpResponse("Labels creating Problem. " + str(e), status=400)
        finally:
            usr.save()
        return HttpResponse("Label created Successfully. ", status=200)


@auth_required
def meta_column_values(request, colName):
    checkRes = usrCheck(request)
    if checkRes["status"] == 0:
        return HttpResponse(checkRes["message"], status=400)
    else:
        usr = checkRes["usrData"]
    adata = usr.getAnndata()
    df = usr.getIntegrationData()
    if colName.lower() == "Cluster".lower():
        colName = "cluster"
    if request.method == "GET":
        if colName in adata.obs_keys() and not np.issubdtype(
            adata.obs[colName].dtype, np.number
        ):
            temp = list(set(adata.obs[colName]))
            if len(temp) == 1:
                return HttpResponse(
                    "Only 1-type value found in the colName: " + colName, status=400
                )
            elif len(temp) > 30:
                return HttpResponse(
                    "More than 30-type values found in the colName: " + colName,
                    status=400,
                )
            temp.sort()
            return JsonResponse(temp, safe=False)
        if colName in df.columns:
            histogram_trace = go.Histogram(
                x=df[colName],
                histnorm="probability density",  # Set histogram normalization to density
                marker_color="rgba(0, 0, 255, 0.7)",  # Set marker color
            )

            # Configure the layout
            layout = go.Layout(
                title="Density Plot for " + colName,  # Set plot title
                xaxis=dict(title=colName),  # Set x-axis label
                yaxis=dict(title="Density"),  # Set y-axis label
            )

            # Create figure
            fig = go.Figure(data=[histogram_trace], layout=layout)

            return HttpResponse(
                base64.b64encode(fig.to_image(format="svg")).decode("utf-8"),
                content_type="image/svg+xml",
            )
        else:
            return HttpResponse("Can't find the colName: " + colName, status=400)
    if request.method == "DELETE":
        col = MetaFileColumn.objects.filter(
            user=request.user, cID=usr.cID, colName=colName
        ).first()
        if col is None:
            return HttpResponse("No such colName called:" + colName, status=400)
        elif col.label == "1":
            return HttpResponse("Please make {colName} inactive first.", status=400)
        MetaFileColumn.objects.filter(
            user=request.user, cID=usr.cID, colName=colName, label="0"
        ).delete()
        return HttpResponse("Delete {colName} Successfully.", status=200)
