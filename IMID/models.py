from django.db import models
import pandas as pd
from .constants import ONTOLOGY, NUMBER_CPU_LIMITS, BUILT_IN_LABELS, BASE_STATIC
from threadpoolctl import threadpool_limits
import pickle
import scanpy as sc
import numpy as np
import os
from django.contrib.auth.models import AbstractUser, Group, Permission
from django.core.files.base import ContentFile
import uuid
from multiprocessing import Process


def get_file_path(instance, filename):
    if hasattr(instance, "user"):
        return os.path.join(
            str(instance.user.uuid), instance.user.username + "_" + filename
        )
    return os.path.join(str(instance.uuid), instance.user.username + "_" + filename)


def get_share_file_path(instance, filename):
    return os.path.join("share", filename)


class CustomUser(AbstractUser):
    uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    groups = models.ManyToManyField(
        Group, related_name="custom_user_groups", blank=True
    )
    user_permissions = models.ManyToManyField(
        Permission, related_name="custom_user_permissions", blank=True
    )

    class Meta:
        verbose_name = "Custom User"
        verbose_name_plural = "Custom Users"


class SharedFileInstance(models.Model):
    user = models.ForeignKey(
        CustomUser, on_delete=models.CASCADE, related_name="share_file_instance"
    )
    file = models.FileField(upload_to=get_share_file_path, null=True)
    label = models.CharField(max_length=50, blank=True, null=True)

    def __str__(self):
        return os.path.basename(self.file.name)

    @property
    def filename(self):
        return os.path.basename(self.file.name)

    @property
    def path(self):
        return self.file.path


class SharedFile(models.Model):
    user = models.ForeignKey(
        CustomUser, on_delete=models.CASCADE, related_name="share_file"
    )
    cohort = models.CharField(max_length=20, blank=True, null=True)
    type1 = models.CharField(
        max_length=10,
        choices=(("meta", "Meta Data"), ("expression", "Expression Data")),
    )
    groups = models.ManyToManyField(
        Group, related_name="shared_file_groups", blank=True
    )
    file = models.ForeignKey(
        SharedFileInstance, on_delete=models.CASCADE, related_name="shared_file"
    )
    label = models.CharField(max_length=200, blank=True, null=True)

    @property
    def filename(self):
        return os.path.basename(self.file.filename)

    class Meta:
        # Define a unique constraint for cohort and type1 fields
        unique_together = ["cohort", "type1"]


class ProcessFile(models.Model):
    user = models.ForeignKey(
        CustomUser, on_delete=models.CASCADE, related_name="process_file"
    )
    cID = models.CharField(max_length=10, blank=False, null=False)
    pickle_file = models.FileField(upload_to=get_file_path, null=True)

    @property
    def filename(self):
        return os.path.basename(self.pickle_file.path)


class UploadedFile(models.Model):
    user = models.ForeignKey(
        CustomUser, on_delete=models.CASCADE, related_name="uploaded_file"
    )
    cID = models.CharField(max_length=10, blank=False, null=False)
    file = models.FileField(upload_to=get_file_path, null=True)
    type1 = models.CharField(max_length=3, blank=False, null=False)
    label = models.CharField(max_length=10, blank=True, null=True)

    @property
    def filename(self):
        return os.path.basename(self.file.path)


class Gene(models.Model):
    id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=50)


class GOTerm(models.Model):
    name = models.CharField(max_length=50, unique=True)
    term = models.CharField(max_length=255, blank=True, null=True)
    gene = models.ManyToManyField(Gene, related_name="go_term")


class MetaFileColumn(models.Model):
    user = models.ForeignKey(
        CustomUser, on_delete=models.CASCADE, related_name="meta_file_column"
    )
    cID = models.CharField(max_length=10, blank=False, null=False)
    colName = models.CharField(max_length=50, blank=False, null=False)
    label = models.CharField(
        max_length=1,
        choices=(("0", "Not Label"), ("1", "Label")),
        blank=False,
        null=False,
    )
    numeric = models.CharField(
        max_length=1,
        choices=(
            ("0", "String type for the field"),
            ("1", "Numeric type for the field"),
        ),
        blank=False,
        null=False,
    )


class userData:
    def __init__(self, cID, uID):
        self.cID = cID
        self.uID = uID
        self.integrationData = pd.DataFrame()  # expression+clinic+label
        self.anndata = None  # expression+clinic for X

    def setIntegrationData(self, df):
        df = df.copy().round(15)
        self.integrationData = df
        self.dfSetAnndata(df)

    def dfSetAnndata(self, df):
        t = df.loc[:, ~(df.columns.isin(BUILT_IN_LABELS))].copy()
        string_cols = t.select_dtypes(include=["object", "bool"]).columns
        for col in string_cols:
            t[col] = t[col].astype("category").cat.codes
        adata = sc.AnnData(np.zeros(t.values.shape), dtype=np.float64)
        adata.X = t.values
        adata.var_names = t.columns.tolist()
        adata.obs_names = df.obs.tolist()
        adata.obs["batch1"] = df.FileName.tolist()
        adata.obs["batch2"] = [
            i + "(" + j + ")" for i, j in zip(df.LABEL.tolist(), df.FileName.tolist())
        ]
        adata.obs["obs"] = df.LABEL.tolist()
        n_comps = 100

        with threadpool_limits(limits=NUMBER_CPU_LIMITS, user_api="blas"):
            sc.tl.pca(
                adata,
                svd_solver="arpack",
                n_comps=min(t.shape[0] - 1, t.shape[1] - 1, n_comps),
            )
        self.anndata = adata

    def setAnndata(self, adata):
        adata = adata.copy()
        if type(adata) == type(sc.AnnData()):
            self.anndata = adata
        elif type(adata) == type(pd.DataFrame()):
            self.dfSetAnndata(adata)

    def setMarkers(self, markers):
        self.anndata.uns["markers"] = markers

    def getMarkers(self, colName="cluster"):
        if "markers" in self.anndata.uns_keys() and colName == "cluster":
            return self.anndata.uns["markers"]
        elif colName in self.anndata.obs_keys():
            adata = self.anndata.copy()
            with threadpool_limits(limits=NUMBER_CPU_LIMITS, user_api="blas"):
                sc.tl.rank_genes_groups(adata, groupby=colName, method="t-test")
                markers = sc.get.rank_genes_groups_df(adata, None)
            return markers
        return None

    def getIntegrationData(self) -> "pd.DataFrame":
        return self.integrationData

    def getAnndata(self) -> "sc.AnnData":
        return self.anndata

    def getCorrectedCSV(self) -> "pd.DataFrame":
        t = self.anndata.to_df().round(12)
        t["FileName"] = self.anndata.obs["batch1"]
        t["obs"] = self.anndata.obs["obs"]
        t["LABEL"] = t["obs"]
        return t

    def getCorrectedClusterCSV(self) -> "pd.DataFrame":
        t = self.anndata.to_df()
        t["FileName"] = self.anndata.obs["batch1"]
        t["obs"] = self.anndata.obs["obs"]
        t["LABEL"] = t["obs"]
        if "cluster" in self.anndata.obs.columns:
            t["cluster"] = self.anndata.obs["cluster"]
            return t
        else:
            return None

    def setFRData(self, xfd):
        self.anndata.obsm["X_umap"] = xfd

    def getFRData(self) -> "pd.DataFrame":
        if "X_umap" in self.anndata.obsm_keys():
            return self.anndata.obsm["X_umap"]
        else:
            return None

    def save(self):
        user_instance = CustomUser.objects.filter(username=self.uID).first()
        if not user_instance:
            return False
        ProcessFile.objects.filter(user=user_instance, cID=self.cID).delete()
        ProcessFile.objects.create(
            user=user_instance,
            cID=self.cID,
            pickle_file=ContentFile(pickle.dumps(self), self.cID + ".pkl"),
        )
        return True

    @classmethod
    def read(self, user, cID) -> "userData":
        user_instance = CustomUser.objects.filter(username=user).first()
        if not user_instance:
            return None
        custom_upload = ProcessFile.objects.filter(user=user_instance, cID=cID).first()
        if not custom_upload:
            return None
        else:
            with open(custom_upload.pickle_file.path, "rb") as f:
                return pickle.loads(f.read())
