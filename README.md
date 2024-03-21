# Patients-RNA-seq-and-Clinic-data-Explorer
Patients RNA-seq and Clinic data Explorer
Gene expression analysis is a powerful tool to gain insight into the mechanisms and processes underlying the biological and phenotypic differences between sample groups and requires highly professional skills for scientists to access, integrate and analyse gene expression. Previously related research focus on establishing tools and platform form single cell RNA-seq analysis, which provide user-friendly graphic interfaces to perform interactive and reproducible gene expression analysis of microarray and RNA-seq datasets and generate a range of publication ready figures. However, those harmonised datasets are not bulked RNA datasets and not patient specific. For some experiments, patient specific samples are collected and cannot be directly applied by this platform. Our platform, provides a solution for performing interactive and reproducible analysis of microarray and RNA-seq bulked gene expression data, enable medical researchers to perform exploratory data analysis and find different/common pattern among patients with similar diseases without informatics proficiency. 
## Data Upload
To test and demonstrate our functionalities, we explored, subsampled and combined four RNA-seq datasets, GSE54456, GSE114284, GSE121212, GSE157194, GSE179633, GSE186476 and GSE130955. 
GSE54456 included gene expression profile from skin samples of 92 psoriatic and 82 normal punch biopsies. The genes were used to quantify gene expression levels. The gene expression was normalized to the number of reads per kilobase per million mapped reads (RPKM).  
GSE114286 included gene expression profile from 9 normal skins from healthy volunteers and 18 lesional skins from patients with psoriasis. The gene expression was normalized through RPKM.  
GSE121212 included gene expression profile of skin tissues obtained from a carefully matched and tightly defined cohort of 28 psoriasis patients, and 38 healthy controls, 27 AD(Atopic dermatitis) patients.  
GSE157194 included 166 skin transcriptomes obtained from 57 patients with moderate to severe AD recruited in the TREAT germany registry between July 2017 and February 2019 at six out of 17 study centres which agreed to participate in the additional and optional bioanalytics module. Intrapersonal lesional (AL) and non-lesional (AN) skin biopsies (4 mm) were collected from 57 patients prior to the initiation of a systemic therapy. Non-lesional samples were taken at least 5 cm from the active lession.  
GSE186476 included 28 skin transcriptomes for Cutaneous lupus erythematosus (CLE) dataset: 14 healthy, 7 lesional skin samples and 7 non-esional skin samples;  
GSE179633 collected 23 skin biopsies of healthy control(HC), DLE (discoid lupus erythematosus, a main type of CLE) and SLE(systemic lupus erythematosus), separated epidermis and dermis and performed single cell RNA sequencing through microfluidics based 10x genomics system;  
For GSE 130955, RNA from skin biopsies from 58 patients in the Prospective Registry for Early Systemic Sclerosis (PRESS) cohort (mean disease duration 1.3 years) and 33 matched healthy controls was examined by nextGen RNA sequencing. 3- or 4-mm diameter punch biopsies were obtained from the forearm skin and immersed in RNAlater solution (Qiagen). RNA was extracted using miRNeasy Mini kits (Qiagen). cDNA libraries were prepared using the Illumina TruSeq stranded Total RNA Library Prep Gold kit, loaded on cBot (Illumina) at a final concentration of 10 pM to perform cluster generation, followed by 2 x 76 bp paired-end sequencing on HiSeq 2500 (Illumina), generating on average around 50 million reads per sample. 
## Reviewing the dataset information 
After clicking the ‘Analyse’ button in the sidebar, information about the study and experiment is displayed in the panel. It shows the first 5 rows for dataset with column names. User will have option to select log2 to transform the dataset.  
## Evaluating the results of EDA 
Several plots are produced as a result of EDA. For example, it displays an interactive corrected 3d plot for the batches after user’s choice of dimension reduction tech such as tsne or umap). User can directly download the corrected data to their own process.  
## Generating and exploring the results of DGEA 
Apart from visualizing the data by batches, users then have option to cluster the dataset by hierarchy clustering, Leiden and HDBSCAN with some parameter tuning. Then top N genes for the clustering method will be generated. Then, LASSO method will be adopted demonstrating the feature importance for the batches and clusters separately.  At the same time, user can select top N(default=20) differential genes in each group. All figures generated and displayed in the user interface, many of which can be explored interactively on the webserver, are publication-ready quality and can be downloaded with a mouse click in PNG format. 
Specifically, users can also select one group in a group clustered by algorithm to find the similarities in that group among different batches. 
## Go enrichment Analysis. 
Users have options to have Go enrichment Analysis. Users can select their interested clustered group and have an analysis. All figures generated and displayed in the user interface, many of which can be explored interactively on the webserver, are publication-ready quality and can be downloaded with a mouse click in PNG format. 


# System Implementation and Package dependence
1. The sys is built based on django+Celery+Redis to avoid some issue of fig.save for matplot.
2. In order to use the Celery and Redis, please have a reference about how to set up it. https://realpython.com/asynchronous-tasks-with-django-and-celery/
```
To install django:
pip install django
python manage.py migrate
python manage.py runserver

To install celery:
pip install celery

To install redis and start the server:
sudo apt install redis
redis-server port 8001

Then open another terminal for test:
redis-cli -p 8001

Note: the reids settings is set up in djangoproject/settings.py with the corresponding port number and serialization method.
Then start the cerlery: 
celery -A djangoproject worker -l info
```
