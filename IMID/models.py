from django.db import models
import pandas as pd
from .constants import ONTOLOGY, BASE_UPLOAD, BASE_STATIC
from threadpoolctl import threadpool_limits
import pickle
import scanpy as sc
import numpy as np
import os

class Gene(models.Model):
    id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=50)


class GOTerm(models.Model):
    name = models.CharField(max_length=50, unique=True)
    term = models.CharField(max_length=255, blank=True, null=True)
    gene = models.ManyToManyField(Gene, related_name="go_term")

class userData():
    def __init__(self,cID,uID):
        self.cID=cID
        self.uID=uID
        self.integrationData = pd.DataFrame() #expression+clinic+label
        self.anndata = None #expression+clinic for X
        self.markers=pd.DataFrame()
        self.fr=None
    
    def setIntegrationData(self, df):
        df=df.round(15)
        self.integrationData=df
        t = df.loc[:, ~(df.columns.isin(["obs", "FileName", "LABEL", "cluster"]))]
        adata = sc.AnnData(np.zeros(t.values.shape), dtype=np.float64)
        adata.X = t.values
        adata.var_names = t.columns.tolist()
        adata.obs_names = df.obs.tolist()
        adata.obs["batch1"] = df.FileName.tolist()
        adata.obs["batch2"] = [
        i + "(" + j + ")" for i, j in zip(df.LABEL.tolist(), df.FileName.tolist())
        ]
        adata.obs['obs']=df.LABEL.tolist()
        with threadpool_limits(limits=2, user_api="blas"):
            sc.tl.pca(adata, svd_solver="arpack", n_comps=100)
        self.anndata=adata
    
    def setAnndata(self,adata):
        self.anndata=adata

    def setMarkers(self,markers):
        self.markers=markers

    def getMarkers(self):
        return self.markers

    def getIntegrationData(self) -> 'pd.DataFrame':
        return self.integrationData
    
    def getAnndata(self) ->'sc.AnnData':
        return self.anndata

    def getCorrectedCSV(self)  -> 'pd.DataFrame':
        t=self.anndata.to_df()
        t['FileName']=self.anndata.obs['batch1']
        t['obs']=self.anndata.obs['obs']
        t['LABEL']=t['obs']
        return t

    def getCorrectedClusterCSV(self)  -> 'pd.DataFrame':
        t=self.anndata.to_df()
        t['FileName']=self.anndata.obs['batch1']
        t['obs']=self.anndata.obs['obs']
        t['LABEL']=t['obs']
        if 'cluster' in self.anndata.obs.columns:
            t['cluster']=self.anndata.obs['cluster']
            return t
        else:
            return None

    def setFRData(self,xfd):
        self.fr=xfd
    
    def getFRData(self)  -> 'pd.DataFrame':
        return self.fr

    def save(self):
        with open(BASE_STATIC+self.uID+'_'+self.cID+'.pkl','wb')as f:
            pickle.dump(self,f)

    @classmethod
    def read(self,uID,cID) -> 'userData':
        if os.path.isfile(BASE_STATIC+uID+'_'+cID+'.pkl'):
            with open(BASE_STATIC+uID+'_'+cID+'.pkl','rb')as f:
                return pickle.load(f)
        else:
            return None