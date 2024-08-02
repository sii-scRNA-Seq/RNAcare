# RNA-care: RNA-based Clinical Analysis and Research Engine
Gene expression analysis is a powerful tool to gain insight into the mechanisms and processes underlying the biological and phenotypic differences between sample groups and requires highly professional skills for scientists to access, integrate and analyse gene expression. Previously related research focus on establishing tools and platform form single cell RNA-seq analysis, which provide user-friendly graphic interfaces to perform interactive and reproducible gene expression analysis of microarray and RNA-seq datasets and generate a range of publication ready figures. However, those harmonised datasets are not bulked RNA datasets and not patient specific. For some experiments, patient specific samples are collected and cannot be directly applied by this platform. Our platform, provides a solution for performing interactive and reproducible analysis of microarray and RNA-seq bulked gene expression data, enable medical researchers to perform exploratory data analysis and find different/common pattern among patients with similar diseases without informatics proficiency. 

# System Implementation and Package dependence
#### 1. Complete code for migration:
To copy RNA-care from GitHub and make it runnable on your local machine, you can follow these steps:
##### Step 1: Clone the Repository

First, clone the repository from GitHub to your local machine.

```bash
git clone https://github.com/sii-scRNA-Seq/RNA-CARE.git
cd RNA-CARE/
```
##### Step 2: Set Up a Virtual Environment(for development, we use python 3.8.19)

It's a good practice to use a virtual environment to manage your project's dependencies.


```bash
# Install virtualenv if you haven't already
pip install virtualenv

# Create a virtual environment
python3 -m virtualenv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```
##### Step 3: Install Dependencies

Install the dependencies listed in the requirements.txt file.

```bash
pip install -r requirements.txt
```
##### Step 4: Configure the Django Settings

Ensure the Django settings file is configured correctly. The default settings for SQLite should be fine if you're running it locally.

Open the settings.py file in your Django project directory and check the database settings:

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / "db.sqlite3",
    }
}
```
You might also want to change the directory for user uploaded files from the default value to some other directory that you have write access to:

```python
MEDIA_ROOT = os.path.join(BASE_DIR, "uploaded")
``` 
##### Step 5: Run Migrations to configure database

In order to create the database `db.sqlite3` and configure the schema, run the following:

```bash
python manage.py migrate
```
Note: It might take a while to run the step `Applying IMID.0003_populate_go_enrich_data`, depending on your internet connection.
##### Step 6: Create a Superuser (if needed)

If you need to create a superuser for accessing the Django admin interface, you can do so with:

```bash
python manage.py createsuperuser
```
##### Step 7: Run the Django Development Server

Finally, run the Django development server to verify that everything is set up correctly.

```bash
python manage.py runserver
```
Open your web browser and go to http://127.0.0.1:8000/ to see your Django project running.

###### Step 7.1 (optional): Upload shared data files in admin interface

For the initialization of the database, superusers have an option to go to the admin interface http://127.0.0.1:8000/admin/ to upload shared datasets to the table:

![image](https://github.com/user-attachments/assets/4390b0bd-3da5-4db6-bc5f-22002611cabc)

Each dataset should include two files, one is Expression Data and the other is Meta Data. They share the same cohort names. SharedFileInstance is used for uploaded shared files management. Then it can be shared among different roles by changing the following info through table SharedFiles:

![image](https://github.com/user-attachments/assets/66367db5-4b61-464e-94de-f40197885f80)

![image](https://github.com/user-attachments/assets/e95f177c-e53e-4b8d-a009-045c8e3f9630)

At the same time, superuser needs to assign group(s) to the specific users, as you see in the following picture, this user was assigned to the ORBIT group. By default, users can see data from all cohorts.

![image](https://github.com/user-attachments/assets/ab348807-7b96-4abb-a7d2-0fc829d0c985)

##### Step 8: Configure Celery and Redis

The project is built based on Django + Celery + Redis to avoid some issues of fig.save for matplotlib. In order to use Celery and Redis, please have a reference about how to set up it: https://realpython.com/asynchronous-tasks-with-django-and-celery/

Install Redis server:
```bash
# On Linux/WSL2(Windows)
sudo apt install redis
# On macOS
brew install redis
```

To start the Redis server, open another console:
```bash
redis-server --port 8001
```
Then open another console for test:
```bash
redis-cli -p 8001
```
If successfully connected, you should see a cursor with `127.0.0.1:8001>`.

Note: Celery settings are defined in `djangoproject/settings.py`, with the corresponding port number and serialization method for redis.

Open another console to start Celery:
```bash
python3 -m celery -A djangoproject worker -l info
```

##### Step 9 (optional): Import the user info from old sys to the new (if needed)
From the old sys folder, run the following to initialise a Django shell:
```bash
python manage.py shell
```
and then:
```python
import pickle
from django.contrib.auth.models import User
usrs=User.objects.all()
with open('/old/sys/user.pkl','wb')as f:
    pickle.dump(usrs,f)
```

Now run the same `python manage.py shell` command from the new sys folder:
```python
from IMID.models import CustomUser
import pickle
with open('/old/sys/user.pkl','rb')as f:
    users=pickle.load(f)
for u in users[1:]:
    CustomUser.objects.create(
        username=u.username,
        email=u.email,
        password=u.password
    )
```

# System Introduction
![image](https://github.com/sii-scRNA-Seq/RNA-CARE/assets/109546311/0b490eb5-67b2-41b2-8e0a-429618f04aab)
Gene expression analysis can be instrumental in comparing gene expression levels in diseased patients versus unaffected counterparts, potentially leading to new treatment strategies. Advances in high-throughput gene expression techniques have significantly increased the number of gene expression studies. Our platform facilitates research on data from IMID (Immune-Mediated Inflammatory Diseases) patients, supported by IMID-Bio-UK and funded by the Medical Research Council. IMID-Bio-UK aims to leverage a rich reserve of biosamples, deeply phenotyped clinical cohorts, and high-quality multi-omic data from various national stratified medicine programs.

We developed a user-friendly webserver for on-the-fly analysis of gene expression data, enabling users without programming skills to process and browse RNA-seq and microarray expression datasets, perform comprehensive gene expression analysis, and generate publication-ready figures and reports. The platform supports combining and harmonizing multiple bulk datasets for integrated analysis.
## Platform Features:
    1. Data Harmonization
    2. Data Transformation
    3. User-Defined and Created Labels
    4. Clustering Process Tracing
    5. Integration of Omics and Clinical Data
    6. Exploratory Data Analysis (EDA)
    7. Differential Gene Expression Analysis (DGEA)
    8. Data Visualization
    9. Gene Enrichment Analysis

# Materials and Methods
## Implementation
Our platform is developed in Python, utilizing several widely used packages. The webserver was built using the Django framework, adhering to the FAIR (Findable, Accessible, Interoperable, and Reusable) principles. The user interface is constructed with a sidebar containing widgets for data collection and transformation options. The platform employs the Plotly graphics system for generating interactive visualizations.
## Workflow overview
    1. Upload Data: Users upload raw count expression data in CSV format along with a meta file describing batch names and clinical conditions.
    2. Data Processing: Options for log2 transformation, batch correction (using methods like Harmony or ComBat), and feature reduction are available.
    3. User-Defined and Created Labels: Users can create and define their own labels based on clinical variables or other criteria, facilitating targeted analysis and personalized insights.
    4. Clustering Process Tracing: Users can trace the entire clustering process, adjusting the interval values between min and max to observe how clustering results evolve and to fine-tune clustering parameters.
    5. Integration of Omics and Clinical Data: The platform enables the integration of multi-omics data with clinical data, providing a comprehensive approach to analyze and interpret complex biological information in conjunction with patient clinical data.
    6. Exploratory Data Analysis: Generate and review interactive 2D plots (e.g., t-SNE, UMAP) based on user selection.
    7. Clustering and Visualization: Perform clustering using methods like Leiden, HDBSCAN, and K-Means, followed by visualizations and differential gene expression analysis.
    8. Gene Plotting: Plot specific genes or candidate genes using violin plots, density plots, or heatmaps.
    9. LASSO Feature Selection: Demonstrate feature importance for specific clusters.
    10. GO Enrichment Analysis: Perform gene ontology analysis for user-selected clusters.
![image](https://github.com/sii-scRNA-Seq/RNA-CARE/assets/109546311/0d0d5aa7-3c79-46a9-acef-e34574a8b2e4)

# Result and demonstrations
## Data collection
For demonstration, we used whole blood samples from the Pathobiology of Early Arthritis Cohort (PEAC), which includes over 350 patients with early inflammatory arthritis. The platform enables users to generate publication-ready figures and interactively explore data.
Integration of Omics and Clinical Data

The main webpage entrance is(assuming you start it at local): http://127.0.0.1:8000/

Our platform supports the integration of multi-omics data with clinical data, facilitating a comprehensive analysis that combines genetic, transcriptomic, proteomic, and metabolomic data with patient-specific clinical information. This integrated approach allows users to uncover more complex and clinically relevant insights. After clicking the ‘Upload’ button in the sidebar, uploaded information about the study and experiment is displayed in the panel. It shows the first 5 rows for dataset with column names. User will have option to select log2 to transform the dataset later. For later calculation.
![image](https://github.com/sii-scRNA-Seq/RNA-CARE/assets/109546311/574b2abe-420d-4053-9e4d-0059576d7ee6)

## Data harmonisation, and transformation
![image](https://github.com/sii-scRNA-Seq/RNA-CARE/assets/109546311/c4d28b6f-f3c1-45c3-9239-6d2686ec3be3)

In this tab, user has option to choose shared cohort(s) to intergrate with, and data processing technique to process the data including removing batch effect, log2 transformation and feature reduction methods.  

## User-Defined and Created Labels
![image](https://github.com/sii-scRNA-Seq/RNA-CARE/assets/109546311/78efc0a9-da7a-4de2-a94e-28bd47f73f82)
![image](https://github.com/sii-scRNA-Seq/RNA-CARE/assets/109546311/bbf5dbd2-4fb5-444a-be6c-610809f6b5d6)

In this section, user can create self-defined fields based on uploaded fields for later analysis as the targeted dependendant variable.

## Exploratory Data Analysis
EDA produces several plots, including interactive 2D plots for dimension reduction techniques like t-SNE and UMAP. Users can download corrected data for further analysis.
![image](https://github.com/sii-scRNA-Seq/RNA-CARE/assets/109546311/4ef55508-69c5-4e19-a8fe-e26f982cdb24)

## Differential Gene Expression Analysis(DGEA)
Apart from visualizing the data by batches, users then have option to cluster the dataset by Kmeans, Leiden and HDBSCAN with some parameter tuning. Then top N genes for the clustering method will be generated. Then, LASSO method will be adopted demonstrating the feature importance for the batches and clusters separately.  At the same time, user can select top N(default=20) differential genes in each group. All figures generated and displayed in the user interface, many of which can be explored interactively on the webserver, are publication-ready quality and can be downloaded with a mouse click in PNG format.
![image](https://github.com/sii-scRNA-Seq/RNA-CARE/assets/109546311/54a144d1-a6e3-4dc5-8eb5-7d502bfee64c)

Besides, we offer a function by which user can easily trace back the procedure of clustering. User just needs to input the min and max value of the range and the expected number of levels. The algorithm will generate clusters based on the number of intervals between the range.
![image](https://github.com/sii-scRNA-Seq/RNA-CARE/assets/109546311/05f7592b-579d-4bed-8bf1-2f22dbce5f85)

## Specifical gene Plots
RNA-care equips users to draw their genes of interest based on the clusters set in the preivous steps, either uploaded or defined through the platform. The system can also give some candidate genes by defined algorims. It provides 3 different gene plots: violin plot, density plot and heatmap plot.
![image](https://github.com/sii-scRNA-Seq/RNA-CARE/assets/109546311/8591e63b-08a4-4ae9-9d44-729e67e3101c)

## Lasso Feature Selection
Specifically, users can also select one group in pre-defined clusters to find the differences v.s other groups
![image](https://github.com/sii-scRNA-Seq/RNA-CARE/assets/109546311/2dbd3c06-34ab-42b5-ae2e-ba8d5faa9522)

## Go enrichment Analysis
Users can perform GO enrichment analysis for specific clusters, enhancing the understanding of biological processes involved.
![image](https://github.com/sii-scRNA-Seq/RNA-CARE/assets/109546311/6ddb9d06-0179-48d5-8959-942b319a2fbc)






