from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path(r"GeneLookup/", views.GeneLookup, name="GeneLookup"),
    path(r"upload/geneExpression/", views.uploadExpression, name="upload_expression"),
    path(r"upload/meta/", views.uploadMeta, name="upload_meta"),
    path(r"eda/integrate/", views.edaIntegrate, name="edaIntegrate"),
    path(r"eda/", views.eda, name="eda"),
    path(r"dega/candiGenes/", views.candiGenes, name="candiGenes"),
    path(r"dega/plot/", views.genePlot, name="genePlot"),
    path(r"dgea/", views.dgea, name="dgea"),
    path(r"cluster/", views.clustering, name="clustering"),
    path(r"cluster/advanced/", views.clusteringAdvanced, name="clusteringAdvanced"),
    path(r"goenrich/", views.goenrich, name="goenrich"),
    path(r"lasso/", views.lasso, name="lasso"),
    path(r"tab/", views.tab, name="tab"),
    path(r"advancedSearch/", views.advancedSearch, name="advancedSearch"),
    path(r"meta/columns/", views.meta_columns, name="meta_columns"),
    path(r"meta/<slug:colName>/", views.meta_column_values, name="meta_columns_values"),
    path(
        r"processedFile/corrected/", views.downloadCorrected, name="downloadCorrected"
    ),
    path(r"user/", views.checkUser, name="checkUser"),
    path(
        r"processedFile/correctedCluster/",
        views.downloadCorrectedCluster,
        name="downloadCorrectedCluster",
    ),
]
