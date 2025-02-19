from django.urls import path
from . import views

urlpatterns = [
    # path("", views.index, name="index"),
    path(r"accounts/rest/login/", views.restLogin, name="restLogin"),
    path(r"GeneLookup/", views.GeneLookup, name="GeneLookup"),
    path(r"geneExpression/", views.opExpression, name="op_expression"),
    path(r"meta/", views.opMeta, name="op_meta"),
    path(r"eda/integrate/", views.edaIntegrate, name="edaIntegrate"),
    path(r"eda/", views.eda, name="eda"),
    path(r"dgea/candiGenes/", views.candiGenes, name="candiGenes"),
    path(r"dgea/plot/", views.genePlot, name="genePlot"),
    path(r"dgea/", views.dgea, name="dgea"),
    path(r"cluster/", views.clustering, name="clustering"),
    path(r"cluster/advanced/", views.clusteringAdvanced, name="clusteringAdvanced"),
    path(r"goenrich/", views.goenrich, name="goenrich"),
    path(r"lasso/", views.lasso, name="lasso"),
    path(r"", views.tab, name="tab"),
    path(r"advancedSearch/", views.advancedSearch, name="advancedSearch"),
    path(r"meta/columns/", views.meta_columns, name="meta_columns"),
    path(
        r"meta/<slug:colName>/",
        views.meta_column_values,
        name="meta_columns_values",
    ),
    path(
        r"processedFile/corrected/", views.downloadCorrected, name="downloadCorrected"
    ),
    path(r"user/", views.checkUser, name="checkUser"),
    path(
        r"processedFile/correctedCluster/",
        views.downloadCorrectedCluster,
        name="downloadCorrectedCluster",
    ),
    path(r"ica/metagenes/", views.downloadICA, name="downloadICA"),
    path(r"ica/", views.ICA, name="ica"),
]
