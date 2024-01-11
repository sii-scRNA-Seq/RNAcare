from django.urls import path
from . import views
urlpatterns=[
	path('', views.index, name='index'),
	path(r'upload/geneExpression/', views.uploadExpression,name="upload_expression"),
	path(r'upload/meta/',views.uploadMeta,name='upload_meta'),
	path(r'eda/',views.eda,name='eda'),
	path(r'dgea/',views.dgea,name='dgea'),
	path(r'cluster/',views.clustering,name='clustering'),
	path(r'cluster/advanced/',views.clusteringAdvanced,name='clusteringAdvanced'),
	path(r'goenrich/',views.goenrich,name='goenrich'),
	path(r'lasso/',views.lasso,name='lasso'),
	path(r'tab/',views.tab,name='tab'),
]
