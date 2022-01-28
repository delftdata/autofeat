# Join Path-Based Data Augmentation for Decision Trees

[![Python 3.7+](https://img.shields.io/badge/python-3.8.2-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![pip](https://img.shields.io/badge/pip-20.0.2-blue.svg)](https://pypi.org/project/pip/)
[![Neo4j Desktop](https://img.shields.io/badge/neo4jDesktop-1.4.10-blue.svg)](https://pypi.org/project/pip/)


## Setup

### neo4j databse
1. Import the database [neo4j-data.dump](neo4j-db/neo4j-data.dump) in neo4j following these [instructions](https://tbgraph.wordpress.com/2020/11/11/dump-and-load-a-database-in-neo4j-desktop/comment-page-1/).


### python setup
1. Create virtual environment

`virtualenv {env-name}`

2. Activate environment 

`source {env-name}/bin/activate`

3. Install requirements 

`pip install -r requirements.txt`

## Run experiments
### Non-Aug baseline 
1. Locate the file [baseline.py](augmentation/baseline.py).
2. Run the script. 

### JoinAll baseline.
1. Locate the file [join-all.py](augmentation/join-all.py).
2. Run the script. 

### BestRank approach. 
1. Locate the file [algorithm_pipeline.py](augmentation/algorithm_pipeline.py).
2. Uncomment line 57: 
`pipeline(datasets, k=1)  # BestRank` 
3. Run the script. 

### Get top-k best ranked join paths
1. Locate the file [algorithm_pipeline.py](augmentation/algorithm_pipeline.py).
2. Uncomment line 58: 
`pipeline(datasets)  # Top-k` 
3. Run the script. 

## Repo structure
1. [augmentation](augmentation) folder

The folder contains the source source and the experiment source code.

2. [neo4j-db](neo4j-db) folder 

Contains the dump of the neo4j database we used for the experiments. 


3. [other-data](other-data) folder contains the splitted datasets. The original source of the datasets:
   * [titanic](https://www.kaggle.com/dmilla/introduction-to-decision-trees-titanic-dataset/data)
   * [kidney-disease](https://www.kaggle.com/akshayksingh/kidney-disease-dataset)
   * [steel-plate-fault](https://www.kaggle.com/bpkapkar/steel-plates-faults-detection?select=Variable+Descriptor.txt)
   * [football](https://www.kaggle.com/estefanytorres/international-football-matches-with-stats-201017?select=FutbolMatches.csv)

4. [plots](plots) folder contains the plots generated in the [Visualisations](Visualisations.ipynb) notebook.
5. [results](results) folder contains the results of the experiments. They are also the source files for the visualisations. 