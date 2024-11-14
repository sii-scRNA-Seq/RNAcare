from functools import lru_cache

from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.base import download_go_basic_obo
from goatools.base import download_ncbi_associations
from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS
from goatools.obo_parser import GODag

from genes_ncbi_proteincoding import GENEID2NT

import random
import string
from .constants import GeneID_URL, NUMBER_CPU_LIMITS
from .models import (
    CustomUser,
    Gene,
    GOTerm,
    userData,
    MetaFileColumn,
    SharedFile,
    UploadedFile,
)
from harmony import harmonize
import pandas as pd
import numpy as np
import scanpy as sc
import collections
from combat.pycombat import pycombat
from matplotlib import pyplot as plt
from django.http import HttpResponse, JsonResponse
from collections import Counter
import requests
import json
import io
import base64
from threadpoolctl import threadpool_limits
import anndata as ad
from sklearn.preprocessing import MinMaxScaler
from django.shortcuts import render
from functools import wraps
import jwt
import datetime
from django.conf import settings
from django.shortcuts import redirect
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference

inference = DefaultInference(n_cpus=NUMBER_CPU_LIMITS)


def auth_required(f):
    @wraps(f)
    def wrap(request, *args, **kwargs):
        is_browser = "Mozilla" in request.META.get("HTTP_USER_AGENT", "")
        if is_browser:
            if not request.user.is_authenticated:
                return redirect("login")
        else:
            token = request.headers.get("Authorization")
            if not token or not token.startswith("Bearer "):
                return JsonResponse(
                    {"error": "Token is missing or invalid"}, status=401
                )
            token = token.split()[1]
            try:
                decoded_token = jwt.decode(
                    token, settings.SECRET_KEY, algorithms=["HS256"]
                )
                request.user = CustomUser.objects.get(
                    username=decoded_token["username"]
                )
            except (
                jwt.ExpiredSignatureError,
                jwt.InvalidTokenError,
                CustomUser.DoesNotExist,
            ):
                return JsonResponse({"error": "Invalid Token"}, status=401)
        return f(request, *args, **kwargs)

    return wrap


def generate_jwt_token(user):
    expiration = datetime.datetime.utcnow() + datetime.timedelta(hours=2)
    token = jwt.encode(
        {"username": user.username, "exp": expiration},
        settings.SECRET_KEY,
        algorithm="HS256",
    )
    return token


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


def preview_dataframe(df):
    if len(df.columns) > 7:
        selected_columns = list(df.columns[:4]) + list(df.columns[-3:])
        df = df[selected_columns]
        df = df.rename(columns={df.columns[3]: "..."})
        df.loc[:, "..."] = "..."
    return df


def has_duplicate_columns(df):
    # Check for duplicated column names
    duplicated_columns = df.columns[df.columns.duplicated()]
    return len(duplicated_columns) > 0


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


def clusteringPostProcess(X2D, adata, method, usr):
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

    adata.obs["cluster"] = li
    usr.setAnndata(adata)

    barChart1 = []
    barChart2 = []

    with plt.rc_context():
        figure1 = io.BytesIO()
        with threadpool_limits(limits=NUMBER_CPU_LIMITS, user_api="blas"):
            sc.tl.rank_genes_groups(adata, groupby=method, method="t-test")
            sc.tl.dendrogram(adata, groupby=method)
            sc.pl.rank_genes_groups_dotplot(
                adata, n_genes=4, show=False, color_map="bwr"
            )
            plt.savefig(figure1, format="png", bbox_inches="tight")
            figure2 = io.BytesIO()
            sc.pl.rank_genes_groups(adata, n_genes=20, sharey=False)
            plt.savefig(figure2, format="png", bbox_inches="tight")
    markers = sc.get.rank_genes_groups_df(adata, None)
    usr.setMarkers(markers)
    usr.save()
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

    return JsonResponse(
        {
            "traces": traces,
            "fileName": base64.b64encode(figure1.getvalue()).decode("utf-8"),
            "fileName1": base64.b64encode(figure2.getvalue()).decode("utf-8"),
            "bc1": barChart1,
            "bc2": barChart2,
        }
    )


def getTopGeneCSV(adata, groupby, n_genes):
    if len(set(adata.obs[groupby])) > 1:
        with threadpool_limits(limits=NUMBER_CPU_LIMITS, user_api="blas"):
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
        if i in text and text[i] is not None:
            retRes.append(text[i])
        else:
            retRes.append(i)
    return retRes


def UploadFileColumnCheck(df, keywords):
    for keyword in keywords:
        if keyword not in df.columns:
            return False, f"No {keyword} column in the expression file"
    if has_duplicate_columns(df):
        return False, "file has duplicated columns."
    return True, ""


def usrCheck(request, flag=1):
    username = request.user.username
    clientID = request.GET.get("cID", None)
    if clientID is None:
        return {"status": 0, "message": "clientID is Required."}
    if flag == 0:
        usr = userData(clientID, username)
    else:
        usr = userData.read(username, clientID)
        if usr is None:
            return {
                "status": 0,
                "message": "Can't find the user/device.Please request from the beginning.",
            }
    return {"status": 1, "usrData": usr}


from rnanorm import CPM


# normalize transcriptomic data
def normalize(df, count_threshold=2000):
    df = df[[col for col in df.columns if not (col.startswith("LOC") and len(col) > 8)]]
    is_rnaseq = df.applymap(lambda x: float(x).is_integer()).mean().mean() > 0.9
    if is_rnaseq:
        df_filtered = df.loc[:, df.sum(axis=0) >= count_threshold]
    else:
        df_filtered = df.copy()
    adata = ad.AnnData(df_filtered)
    adata.obs.index = [i for i in df_filtered.index]

    if is_rnaseq:
        adata.var["mt"] = adata.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
        # Filter cells with >5% mitochondrial genes
        adata = adata[adata.obs.pct_counts_mt < 5, :]

        # sc.pp.normalize_total(adata, target_sum=1e6)
        # adata.X = calculate_deseq2_normalization(adata.to_df()) # the result is the same with running CPM()
        adata.X = CPM().fit_transform(adata.to_df())
    # counts = adata.to_df()
    # metadata = pd.DataFrame(
    #    {"condition": ["sc1"] * counts.shape[0]}, index=counts.index
    # )
    # metadata["condition"][0] = "sc2"
    # dds = DeseqDataSet(
    #    counts=counts,
    #    metadata=metadata,
    #    design_factors="condition",
    #    inference=inference,
    # )
    # dds.deseq2()
    # rdf = dds.to_df()
    return adata


# normalize clinic data
def normalize1(df, log2="No"):
    # Identify numeric columns
    numeric_columns = df.select_dtypes(include=np.number).columns
    string_columns = df.select_dtypes(exclude=np.number).columns

    # Separate numeric and string data
    df_numeric = df[numeric_columns]
    df_strings = df[string_columns]

    # scaler = MinMaxScaler()

    # Fit and transform the numeric data
    if df_numeric.shape[1] > 0:
        if log2 == "Yes":
            df_numeric_normalized = np.log1p(df_numeric)
        else:
            scaler = MinMaxScaler()
            df_numeric_normalized = pd.DataFrame(
                scaler.fit_transform(df_numeric), columns=numeric_columns
            )
        df_numeric_normalized.index = df_numeric.index
        df_normalized = pd.concat([df_strings, df_numeric_normalized], axis=1)
    else:
        df_normalized = df_strings
    df_normalized.dropna(axis=0, inplace=True)

    return df_normalized


def loadSharedData(request, integrate, cID):
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
    return files, files_meta


def integrateCliData(request, integrate, cID, files_meta):
    f = UploadedFile.objects.filter(user=request.user, type1="cli", cID=cID).first()
    if f is not None:
        temp0 = pd.read_csv(f.file.path)
        temp0 = temp0.dropna(axis=1)
    else:
        temp0 = pd.DataFrame()
    if integrate[0] != "null" or integrate != [""]:  # jquery plugin compatible
        for i in files_meta:
            if temp0.shape == (0, 0):
                temp0 = pd.read_csv(i).dropna(axis=1, inplace=False)

            else:
                temp0 = pd.concat(
                    [temp0, pd.read_csv(i).dropna(axis=1, inplace=False)],
                    axis=0,
                    join="inner",
                )
    return temp0


def integrateExData(files, temp0, log2, corrected):
    dfs = []
    batch = []
    obs = []
    for file in files:
        temp01 = pd.read_csv(file).set_index("ID_REF")
        # filter records that have corresponding clinical data in meta files.
        temp = temp01.join(temp0[["ID_REF", "LABEL"]].set_index("ID_REF"), how="inner")
        temp1 = temp.reset_index().drop(
            ["ID_REF", "LABEL"], axis=1, inplace=False
        )  # exclude ID_REF & LABEL
        temp1.index = temp.index
        # exclude NA
        temp1.dropna(axis=1, inplace=True)
        if temp1.shape[0] != 0:
            temp1 = normalize(temp1)
            dfs.append(temp1)
            # color2.extend(list(temp.LABEL))
            batch.extend(
                ["_".join(file.split("_")[1:]).split(".csv")[0]]
                * temp1.to_df().shape[0]
            )
            # obs.extend(temp.ID_REF.tolist())
            obs.extend(temp1.to_df().index.tolist())
    if log2 == "Yes":
        dfs = [np.log2(i.to_df() + 1) for i in dfs]
    else:
        dfs = [i.to_df() for i in dfs]

    dfs1 = None
    if len(dfs) > 1:
        if corrected == "Combat":
            dfs1 = combat([i.T for i in dfs])
        elif corrected == "Harmony":
            dfs1 = harmony(dfs, batch, obs)
        elif corrected == "BBKNN":
            dfs1 = bbknn(dfs)
    elif len(dfs) == 0:
        return None
    else:
        dfs1 = dfs[0]

    dfs1["ID_REF"] = obs
    dfs1["FileName"] = batch
    return dfs1


"""flag==1 means response after uploading; flag==0 means initialize for the tab view."""


def getExpression(request, cID, flag=1):
    context = {}
    for file in UploadedFile.objects.filter(
        user=request.user, type1="exp", cID=cID
    ).all():
        df = pd.read_csv(file.file.path, nrows=5, header=0)
        f = file.file.name.split("/")[-1]

        check, mess = UploadFileColumnCheck(df, ("ID_REF",))
        if check is False:
            return HttpResponse(mess, status=400)
        df = preview_dataframe(df)
        json_re = df.reset_index().to_json(orient="records")
        data = json.loads(json_re)
        context["_".join(f.split("_")[1:])] = {}
        context["_".join(f.split("_")[1:])]["d"] = data
        context["_".join(f.split("_")[1:])]["names"] = ["index"] + df.columns.to_list()
    if flag == 0 and len(context) == 0:
        return HttpResponse("", status=200)
    return render(request, "table.html", {"root": context})


def getMeta(request, cID, flag=1):
    context = {}
    f = UploadedFile.objects.filter(user=request.user, type1="cli", cID=cID).first()
    if f is None:
        if flag == 1:
            return HttpResponse(mess, status=400)
        else:
            return HttpResponse("", status=200)
    context["metaFile"] = {}
    df = pd.read_csv(f.file.path, nrows=5, header=0)
    check, mess = UploadFileColumnCheck(df, ("ID_REF", "LABEL"))
    if check is False:
        return HttpResponse(mess, status=400)
    df = preview_dataframe(df)
    json_re = df.reset_index().to_json(orient="records")
    data = json.loads(json_re)
    context["metaFile"]["d"] = data
    context["metaFile"]["names"] = ["index"] + df.columns.to_list()
    return render(request, "table.html", {"root": context})
