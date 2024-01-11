from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from django.contrib.auth.decorators import login_required
import pandas as pd
import json
import os,glob
import sys
from sklearn.manifold import TSNE
import umap.umap_ as umap
import scanpy as sc
import numpy as np
from matplotlib import pyplot as plt
import hdbscan
import random
import string
from subprocess import Popen
import collections
import scanpy.external as sce

sys.path.append('/home/mt229a/Downloads/')#gene_result.txt, genes_ncbi_proteincoding.py, go-basic.obo

import matplotlib
matplotlib.use('agg')
import plotly.graph_objects as go
# Create your views here.

from genes_ncbi_proteincoding import GENEID2NT
from goatools.base import download_go_basic_obo
from goatools.base import download_ncbi_associations
from goatools.obo_parser import GODag
from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS

BASE_UPLOAD='IMID/geneData/upload/'
BASE_STATIC='IMID/static/temp/'
#lasso.R for data visualization


@login_required()
def index(request):
	return render(request,'index.html',)

@login_required()
def tab(request):
	return render(request,'tab.html',)

@login_required()	
def uploadExpression(request):
	if request.method!='POST':
		return HttpResponse('Method not allowed',status=405)
	username=request.user.username
	for file in glob.glob(BASE_UPLOAD+'/'+username+'*'):
		os.remove(file)		
	fileNames=[]
	context={}
	files=request.FILES.getlist('files[]',None)
	for f in files:
		handle_uploaded_file1(f,username)
		fileNames.append(BASE_UPLOAD+username+'_'+f.name)
		context[f.name]={}
		
	for f in fileNames:
		df=pd.read_csv(f).head()
		if len(df.columns)>7:#only show 7 columns
			df=pd.concat([df.iloc[:,0:4],df.iloc[:,-3:]],axis=1)

			temp=df.columns[3]
			df.rename(columns={temp:'...'},inplace=True)
			df['...']='...'
		json_re=df.reset_index().to_json(orient='records')
		data=json.loads(json_re)
		context['_'.join(f.split('_')[1:])]['d']=data
		context['_'.join(f.split('_')[1:])]['names']=['index']+df.columns.to_list()	
	return render(request,'table.html',{'root':context})
	
def uploadMeta(request):
	if request.method!='POST':
		return HttpResponse('Method not allowed',status=405)
	username=request.user.username
	files=request.FILES.getlist('meta',None)[0]
	handle_uploaded_file1(files,username,'meta')
	f=BASE_UPLOAD+username+'_meta.csv'
	context={}
	context['metaFile']={}
	df=pd.read_csv(f).head()
	if len(df.columns)>7:#only show 7 columns
		df=pd.concat([df.iloc[:,0:4],df.iloc[:,-3:]],axis=1)

		temp=df.columns[3]
		df.rename(columns={temp:'...'},inplace=True)
		df['...']='...'
	json_re=df.reset_index().to_json(orient='records')
	data=json.loads(json_re)
	context['metaFile']['d']=data
	context['metaFile']['names']=['index']+df.columns.to_list()
	return render(request,'table.html',{'root':context})
	
@login_required()	
def eda(request):
	username=request.user.username
	corrected=request.GET.get('correct','Combat')
	log2=request.GET.get('log2','No')
	fr=request.GET.get('fr','TSNE')
	integrate=request.GET.get('integrate','').split(',')
	if '' in integrate:
		integrate.remove('')
	for file in glob.glob(BASE_STATIC+'/'+username+'*'):
		os.remove(file)
		
	directory=os.listdir(BASE_UPLOAD)
	files=[i for i in directory if username ==i.split('_')[0]]
	
	in_ta={'SERA':'share_SERA_BLOOD.csv','PEAC':'share_PEAC_recon.csv','PSORT':'share_PSORT.csv','RAMAP':'share_RAMAP_WHL.csv'}
	for i in integrate:
		if i in in_ta:
			files.append(in_ta[i])
	dfs=[]
	batch=[]
	obs=[]
	temp0=[]
	color2=[]
	flag=0
	
	temp0=pd.read_csv(BASE_UPLOAD+username+'_meta.csv')
	if len(integrate)!=0:
		temp0=pd.concat([temp0,pd.read_csv(BASE_UPLOAD+'share_meta.csv')],axis=0,join='inner')
	for file in files:
		if 'meta' in file:
			flag=1
			continue
		temp01=pd.read_csv(BASE_UPLOAD+file).set_index('ID_REF')
		if 'raw' in file or 'RAW' in file:
			temp01= temp01.div(temp01.sum(axis=1),axis=0)*1e6
			#temp01=temp01/temp01.sum()*1e6
		temp=temp01.join(temp0[['ID_REF','LABEL']].set_index('ID_REF'),how='inner')
		temp1=temp.reset_index().drop(['ID_REF','LABEL'],axis=1,inplace=False)#exclude ID_REF & LABEL
		#temp0.append(temp1.filter(regex='c_'))#clinic data
		#temp1=temp1.loc[:,~temp1.columns.str.startswith('c_')]#transcript data
		#rpkm to CPM
		if temp1.shape[0]!=0:
			temp1=(temp1.div(temp1.sum(axis=0), axis=1) * 1e6) / temp1.sum().sum()
			#exclude NA
			temp1=temp1.dropna(axis=1)
			dfs.append(temp1)
			
			#color2.extend(list(temp.LABEL))
			batch.extend(['_'.join(file.split('_')[1:]).split('.csv')[0]]*temp1.shape[0])
			#obs.extend(temp.ID_REF.tolist())
			obs.extend(temp.index.tolist())
		
	if log2=='Yes':
		dfs=[np.log2(i+1) for i in dfs]

	dfs1=None
	if len(dfs)>1:
		if corrected=='Combat':
			dfs1=combat([i.T for i in dfs])
		elif corrected=='Harmony':
			dfs1=harmony(dfs,batch,obs)
		elif corrected=='BBKNN':
			dfs1=bbknn(dfs)
	else:
		dfs1=dfs[0]
	dfs1['ID_REF']=obs
	dfs1['FileName']=batch
	#temp0=pd.concat(temp0,axis=0).reset_index(drop=True) #combine all clinic data
	if flag==0:
		return HttpResponse("Can't find meta file",status=400)
	
	temp=dfs1.set_index('ID_REF').join(temp0.set_index('ID_REF'),how='inner')
	temp['obs']=temp.index.tolist()
	#temp['FileName']=batch#inner join may not match so valued beforehand
	temp.to_csv(BASE_STATIC+username+'_corrected.csv',index=False)

	color2=[i+'('+j+')' for i,j in zip(temp.LABEL,temp.FileName)]
		
	dfs1.drop(['ID_REF'],axis=1,inplace=True)
	dfs1.drop(['FileName'],axis=1,inplace=True)
	
	if fr=='TSNE':
		tsne=TSNE(n_components=3,random_state=42)
		X3D1=tsne.fit_transform(dfs1)
	else:
		
		umap1=umap.UMAP(n_components=3,random_state=42)
		X3D1=umap1.fit_transform(dfs1)
	
	with open(BASE_STATIC+username+'_fr.json','w')as f:
		f.write(json.dumps(X3D1.tolist()))
		
	traces=zip_for_vis(X3D1.tolist(),temp.FileName,temp.obs)
	traces1=zip_for_vis(X3D1.tolist(),color2,temp.obs)
	context={'dfs1':json.dumps(traces),'dfs2':json.dumps(traces1),'fr':fr,'log':log2,'correct':corrected}
	#return render(request,'eda.html',context)
	return JsonResponse(context)

def zip_for_vis(X3D1,batch,obs):
	traces={}
	for i,j,k in zip(X3D1,batch,obs):
		j=str(j)
		if j not in traces:
			traces[j]={'data':[i],'obs':[k]}
		else:
			traces[j]['data'].append(i)
			traces[j]['obs'].append(k)
	return traces
	
@login_required()
def dgea(request):
	username=request.user.username
	clusters=request.GET.get('clusters','false')
	df=pd.read_csv(BASE_STATIC+username+'_corrected.csv')
	t=df.loc[:,~(df.columns.isin(['obs','FileName', 'LABEL']))]#'obs','FileName', 'LABEL'
	adata = sc.AnnData(np.zeros(t.values.shape),dtype=np.float64)
	adata.X=t.values
	adata.var_names=t.columns.tolist()
	adata.obs_names=df.obs.tolist()
	adata.obs['batch1']=df.FileName.tolist()
	adata.obs['batch2']=[i+'('+j+')' for i,j in zip(df.LABEL.tolist(),df.FileName.tolist())]
	sc.tl.pca(adata, svd_solver='arpack')
	adata.write(BASE_STATIC+username+'_adata.h5ad')
	random_str=get_random_string(8)
	if clusters=='false':	
		with plt.rc_context():
			if len(set(adata.obs['batch1']))>1:
				sc.tl.rank_genes_groups(adata, groupby='batch1', method='t-test')
				sc.tl.dendrogram(adata, groupby='batch1')
				sc.pl.rank_genes_groups_dotplot(adata, n_genes=4, show=False)
				plt.savefig(BASE_STATIC+username+'_batch1_'+random_str+'.png',bbox_inches='tight')
			else:
				pass
			if len(set(adata.obs['batch2']))>1:
				sc.tl.rank_genes_groups(adata, groupby='batch2', method='t-test')
				sc.tl.dendrogram(adata, groupby='batch2')
				sc.pl.rank_genes_groups_dotplot(adata, n_genes=4, show=False)
				plt.savefig(BASE_STATIC+username+'_batch2_'+random_str+'.png',bbox_inches='tight')
			else:
				pass
		return JsonResponse([username+'_batch1_'+random_str+'.png',username+'_batch2_'+random_str+'.png'],safe=False)
	if clusters=='true':
		#show top gene for specific group
		pass

@login_required()
def clustering(request):
	username=request.user.username
	cluster=request.GET.get('cluster','LEIDEN')
	param=request.GET.get('param',None)
	
	adata=sc.read(BASE_STATIC+username+'_adata.h5ad')
	random_str=get_random_string(8)
	df=pd.read_csv(BASE_STATIC+username+'_corrected.csv')
	with open(BASE_STATIC+username+'_fr.json','r')as f:
		X3D1=json.loads(f.read())
	if cluster=='LEIDEN':
		if param is None:
			param=1
		sc.pp.neighbors(adata, n_neighbors=40, n_pcs=40)
		sc.tl.leiden(adata,resolution=float(param))
		if len(set(adata.obs['leiden']))==1:
			#throw error for just 1 cluster
			return HttpResponse('Only 1 Cluster after clustering',status=400)
		
		df['cluster']=[i for i in adata.obs['leiden']]
		df.to_csv(BASE_STATIC+username+'_corrected_clusters.csv',index=False)
		
		traces=zip_for_vis(X3D1,list(adata.obs['leiden']),adata.obs_names.tolist())
		adata.write(BASE_STATIC+username+'_adata.h5ad')
		with plt.rc_context():
			sc.tl.rank_genes_groups(adata, groupby='leiden', method='t-test')
			sc.tl.dendrogram(adata, groupby='leiden')
			sc.pl.rank_genes_groups_dotplot(adata, n_genes=4, show=False)
			plt.savefig(BASE_STATIC+username+'_cluster_'+random_str+'_1.png',bbox_inches='tight')
			sc.pl.rank_genes_groups(adata,n_genes=20,sharey=False)
			plt.savefig(BASE_STATIC+username+'_cluster_'+random_str+'_2.png',bbox_inches='tight')
			markers=sc.get.rank_genes_groups_df(adata,None)
			markers.to_csv(BASE_STATIC+username+'_markers.csv',index=False)
		b=adata.obs.sort_values(['batch1','leiden']).groupby(['batch1','leiden']).count().reset_index()
		#print(b)
		b=b[['batch1','leiden','batch2']]
		b.columns=['batch','cluster','count']
		barChart1=[{'x':sorted(list(set(b['cluster'].tolist()))),'y':b[b['batch']==i]['count'].tolist(),'name':i,'type':'bar'} for i in set(b['batch'].tolist())]
		#print(barChart1)
		
		b=adata.obs.sort_values(['batch2','leiden']).groupby(['batch2','leiden']).count().reset_index()
		b=b[['batch2','leiden','batch1']]
		b.columns=['batch','cluster','count']
		barChart2=[{'x':sorted(list(set(b['cluster'].tolist()))),'y':b[b['batch']==i]['count'].tolist(),'name':i,'type':'bar'} for i in set(b['batch'].tolist())]
		
		return JsonResponse({'traces':traces,'fileName':username+'_cluster_'+random_str+'_1.png','fileName1':username+'_cluster_'+random_str+'_2.png','bc1':barChart1,'bc2':barChart2})
	elif cluster=='HDBSCAN':
		if param is None:
			param=20
		if int(param)<=5:
			param=5
		labels=hdbscan.HDBSCAN(min_cluster_size=int(param)).fit_predict(adata.X)		
		#freq=collections.Counter(labels)
		#m=[freq[key] for key in freq]
		#m.sort()
		#if m[0]<=5:
		#	for i in range(len(m)):
		#		if sum(m[:(i+1)])<=5:
		#			continue
		#		labels=hdbscan.HDBSCAN(min_cluster_size=(m[i]+1)).fit_predict(adata.X)
		#		break
		if len(set(labels))==1:
			#throw error for just 1 cluster
			return HttpResponse('Only 1 Cluster after clustering',status=400)
		labels=[str(i+1) for i in labels]
		adata.obs['hdbscan']=labels
		adata.obs['hdbscan']=adata.obs['hdbscan'].astype('category')
		adata.write(BASE_STATIC+username+'_adata.h5ad')
		
		df['cluster']=[i for i in adata.obs['hdbscan']]
		df.to_csv(BASE_STATIC+username+'_corrected_clusters.csv',index=False)
		
		traces=zip_for_vis(X3D1,labels,adata.obs_names.tolist())
		
		with plt.rc_context():
			sc.tl.rank_genes_groups(adata, groupby='hdbscan', method='t-test')
			sc.tl.dendrogram(adata, groupby='hdbscan')
			sc.pl.rank_genes_groups_dotplot(adata, n_genes=4, show=False)
			plt.savefig(BASE_STATIC+username+'_cluster_'+random_str+'_1.png',bbox_inches='tight')
			sc.pl.rank_genes_groups(adata,n_genes=20,sharey=False)
			plt.savefig(BASE_STATIC+username+'_cluster_'+random_str+'_2.png',bbox_inches='tight')
			markers=sc.get.rank_genes_groups_df(adata,None)
			markers.to_csv(BASE_STATIC+username+'_markers.csv',index=False)
				
		b=adata.obs.sort_values(['batch1','hdbscan']).groupby(['batch1','hdbscan']).count().reset_index()
		b=b[['batch1','hdbscan','batch2']]
		b.columns=['batch','cluster','count']
		barChart1=[{'x':sorted(list(set(b['cluster'].tolist()))),'y':b[b['batch']==i]['count'].tolist(),'name':i,'type':'bar'} for i in set(b['batch'].tolist())]
		
		b=adata.obs.sort_values(['batch2','hdbscan']).groupby(['batch2','hdbscan']).count().reset_index()
		b=b[['batch2','hdbscan','batch1']]
		b.columns=['batch','cluster','count']
		barChart2=[{'x':sorted(list(set(b['cluster'].tolist()))),'y':b[b['batch']==i]['count'].tolist(),'name':i,'type':'bar'} for i in set(b['batch'].tolist())]
			
		return JsonResponse({'traces':traces,'fileName':username+'_cluster_'+random_str+'_1.png','fileName1':username+'_cluster_'+random_str+'_2.png','bc1':barChart1,'bc2':barChart2})


@login_required()
def clusteringAdvanced(request):
	if request.method=='GET' and 'cluster' not in request.GET:
		return render(request,'clustering_advance.html',)
	else:
		username=request.user.username
		cluster=request.GET.get('cluster','LEIDEN')
		minValue=float(request.GET.get('min','0'))
		maxValue=float(request.GET.get('max','1'))
		level=int(request.GET.get('level',3))
		adata=sc.read(BASE_STATIC+username+'_adata.h5ad')
		df=pd.read_csv(BASE_STATIC+username+'_corrected.csv')[['LABEL']]
		if level>10 or level<=1:
			return HttpResponse('Error for the input',status=400)
		if cluster=='LEIDEN':
			if minValue<=0:
				minValue=0
			if maxValue>=2:
				maxValue=2
			for i,parami in enumerate(np.linspace(minValue,maxValue,level)):
				sc.pp.neighbors(adata, n_neighbors=40, n_pcs=40)
				sc.tl.leiden(adata,resolution=float(parami))
				df['level'+str(i+1)]=['level'+str(i+1)+'_'+str(j) for j in adata.obs['leiden']]
		elif cluster=='HDBSCAN':
			if minValue<=5:
				minValue=5
			if maxValue>=100:
				maxValue=100
			for i,parami in enumerate(np.linspace(minValue,maxValue,level)):
				df['level'+str(i+1)]=['level'+str(i+1)+'_'+str(j) for j in hdbscan.HDBSCAN(min_cluster_size=int(parami)).fit_predict(adata.X)]
				
		result=fromPdtoSangkey(df)
		return JsonResponse(result)


def fromPdtoSangkey(df):
	df=df.copy()
	source=[]
	df[['parent_level']]=[i+' ' for i in df[['LABEL']].values]
	columns=df.columns.tolist()

	for i in columns:
		source.extend([j[0] for j in df[[i]].values])
	nodes=collections.Counter(source)
	
	source={}
	for i in range(1,len(columns)):
		agg=df.groupby([columns[i-1],columns[i]]).size().reset_index()
		agg.columns=[columns[i-1],columns[i],'count']
		for index,row in agg.iterrows():
			source[(row[columns[i-1]],row[columns[i]])]=row['count']
			
	result={}
	result1=[]
	dic_node={}
	for i,j in enumerate(nodes.keys()):
		result1.append({'node':i,'name':j})
		dic_node[j]=i#name->number
	result['nodes']=result1
	
	result1=[]
	for i in source:
		result1.append({'source':dic_node[i[0]],'target':dic_node[i[1]],'value':source[i]})
	result['links']=result1
	#return json.dumps(result)
	
	sou=[]
	tar=[]
	value=[]
	result={}
	for i in result1:
		sou.append(i['source'])
		tar.append(i['target'])
		value.append(i['value'])
	result['source1']=sou
	result['target1']=tar
	result['value1']=value
	result['label']=list(nodes.keys())
	return result
	
		
		

obo_fname=download_go_basic_obo()
fin_gene2go=download_ncbi_associations()
obodag=GODag("go-basic.obo")	
mapper={}
for key in GENEID2NT:
	mapper[GENEID2NT[key].Symbol]=GENEID2NT[key].GeneID   
inv_map={v:k for k,v in mapper.items()}
objanno=Gene2GoReader(fin_gene2go,taxids=[9606])
ns2assoc=objanno.get_ns2assc()#bp,cc,mf
	
goeaobj=GOEnrichmentStudyNS(GENEID2NT.keys(),ns2assoc,obodag,propagate_counts=False,alpha=.05,methods=['fdr_bh'])
GO_items=[]
temp=goeaobj.ns2objgoea['BP'].assoc#BIOLOGICAL_PROCESS
for item in temp:
	GO_items+=temp[item]
temp=goeaobj.ns2objgoea['CC'].assoc#MOLECULAR_FUNCTION
for item in temp:
	GO_items+=temp[item]     
temp=goeaobj.ns2objgoea['MF'].assoc#CELLULAR COMPONENT
for item in temp:
	GO_items+=temp[item]
			
@login_required()
def goenrich(request):
	username=request.user.username
	cluster_n=request.GET.get('cluster_n',0)
	random_str=get_random_string(8)
	
	df=pd.read_csv(BASE_STATIC+username+'_corrected.csv')
	if any(df.columns.str.startswith('c_')) is True or len(set(df.columns).intersection({'age','crp','bmi','esr','BMI'}))>0 :
		return HttpResponse('Not Allowed Clinic Data',status=400)
	markers=pd.read_csv(BASE_STATIC+username+'_markers.csv')
	markers=markers[(markers.pvals_adj<0.05)&(markers.logfoldchanges>.5)&(markers.group==int(cluster_n))]
	if len(markers.index)==0:
		return HttpResponse('No marker genes',status=400)	
	df=go_it(markers.names.values)
	df['per']=df.n_genes/df.n_go
	df1=df.groupby('class').head(10).reset_index(drop=True)
	
	fig = go.Figure()
	fig.add_trace(go.Bar(
		y=df1[df1['class'] =='molecular_function'].term[::-1],
		x=df1[df1['class'] =='molecular_function'].per[::-1],
		name='molecular_function',
		customdata=['P_corr='+str(round(i,5)) for i in df1[df1['class'] =='molecular_function'].p_corr[::-1]],
		hovertemplate = "Ratio: %{x:.5f}<br> %{customdata}",
		orientation='h',
		marker={
        		'color':df1[df1['class'] =='molecular_function'].p_corr[::-1],
        		'colorscale':'oranges_r',
        		}
        ))

	fig.add_trace(go.Bar(
		y=df1[df1['class'] =='cellular_component'].term[::-1],
		x=df1[df1['class'] =='cellular_component'].per[::-1],
		name='cellular_component',
		customdata=['P_corr='+str(round(i,5)) for i in df1[df1['class'] =='cellular_component'].p_corr[::-1]],
		hovertemplate = "Ratio: %{x:.5f}<br> %{customdata}",
		orientation='h',
		marker=dict(
			color=df1[df1['class'] =='cellular_component'].p_corr[::-1],
			colorscale='purp_r'
		)
	))
	
	fig.add_trace(go.Bar(
    		y=df1[df1['class'] =='biological_process'].term[::-1],
    		x=df1[df1['class'] =='biological_process'].per[::-1],
    		name='biological_process',
    		customdata=['P_corr='+str(round(i,5)) for i in df1[df1['class'] =='biological_process'].p_corr[::-1]],
    		hovertemplate = "Ratio: %{x:.5f}<br> %{customdata}",
    		orientation='h',
    		marker=dict(
        		color=df1[df1['class'] =='biological_process'].p_corr[::-1],
        		colorscale='blues_r'
    		)
	))

	fig.update_layout(barmode='stack',height=1200,
		uniformtext_minsize=50, uniformtext_mode='hide',xaxis= dict(title= 'Gene Ratio'))
		
	fig.write_image(BASE_STATIC+username+'_goenrich_'+random_str+'.png')
	#return render(request,'goenrich.html',{'fileName':username+'_goenrich_'+random_str+'.png'})
	return JsonResponse({'fileName':username+'_goenrich_'+random_str+'.png'})

@login_required()
def lasso(request):
	username=request.user.username
	random_str=get_random_string(8)
	cluster=int(request.GET.get('cluster_n',0))+1
	stdout_file=BASE_STATIC+username+'_'+random_str+'_stdout.txt'
	stderr_file=BASE_STATIC+username+'_'+random_str+'_stderr.txt'
	code=400
	#print('Rscript lasso.R '+username+' '+str(cluster))
	try:
		with open(stdout_file,'w')as out, open(stderr_file,'w')as err:
			Popen(['Rscript lasso.R '+username+' '+str(cluster)+' '+random_str],stdout=out,stderr=err,shell=True).communicate()#calling Rscript
		with open(stdout_file,'r')as out, open(stderr_file,'r')as err:
			stdout=out.read()
			stderr=err.read()
		code=200
	except:
		retValueOut=str(stderr)
		code=400
	finally:
		if code==200:
			if os.path.exists(stdout_file):
				os.remove(stdout_file)
			if os.path.exists(stderr_file):
				os.remove(stderr_file)
		else:
			return HttpResponse('Error happened on the Server',status=code)
	return JsonResponse({'fileName':username+'_'+random_str+'_lasso.png'})
	
	
def go_it(test_genes):
	global mapper,goeaobj,inv_map,GO_items
	mapped_genes=[]
	for gene in test_genes:
		if gene in mapper:
			mapped_genes.append(mapper[gene])
	goea_results_all=goeaobj.run_study(mapped_genes)
	goea_result_sig=[r for r in goea_results_all if r.p_fdr_bh<.05]
	GO=pd.DataFrame(list(map(lambda x:[x.GO, x.goterm.name, x.goterm.namespace, x.p_uncorrected, x.p_fdr_bh, x.ratio_in_study[0],x.ratio_in_study[1],GO_items.count(x.GO),list(map(lambda y:inv_map[y],x.study_items))],goea_result_sig)),columns=['GO','term','class','p','p_corr','n_genes','n_study','n_go','study_genes'])

	GO=GO[GO.n_genes>1]
	return GO
		
def get_random_string(length):
	letters=string.ascii_lowercase
	result_str = ''.join(random.choice(letters) for i in range(length))
	return result_str
	
def handle_uploaded_file1(f,username,filename=''):
	if filename=='':
		with open(BASE_UPLOAD+username+'_'+f.name,'wb+') as dest:
			for chunk in f.chunks():
				dest.write(chunk)
	else:
		with open(BASE_UPLOAD+username+'_'+filename+'.csv','wb+') as dest:
			for chunk in f.chunks():
				dest.write(chunk)

from combat.pycombat import pycombat
def combat(dfs):
	df_exp=pd.concat(dfs,join='inner',axis=1)
	batch=[]
	datasets=dfs
	for j in range(len(datasets)):
		batch.extend([j for _ in range(len(datasets[j].columns))]) 
	df_corrected=pycombat(df_exp,batch)
	Xc=df_corrected.T
	return Xc.reset_index(drop=True)

from harmony import harmonize 
#pip install harmony-pytorch 
def harmony(dfs,batch,obs):
	dfs1=pd.concat(dfs,join='inner',axis=0)
	adata = sc.AnnData(np.zeros(dfs1.values.shape),dtype=np.float64)#'obs','FileName', 'LABEL'
	adata.X=dfs1.values
	adata.var_names=dfs1.columns.tolist()
	adata.obs_names=obs
	adata.obs['batch']=batch
	#sc.tl.pca(adata)
	#sce.pp.harmony_integrate(adata, 'batch')
	#ha=adata.obsm['X_pca_harmony']
	#return pd.DataFrame(ha,columns=['PCA_'+str(i+1) for i in range(ha.shape[1])]).reset_index(drop=True)
	ha=harmonize(adata.X, adata.obs, batch_key = 'batch')
	return pd.DataFrame(ha,columns=dfs1.columns).reset_index(drop=True)
	
def bbknn(dfs):
	return dfs
