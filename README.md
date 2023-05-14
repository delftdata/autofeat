# Transitive Feature Discovery over Join Paths

[![Python 3.7+](https://img.shields.io/badge/python-3.8.2-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![pip](https://img.shields.io/badge/pip-20.0.2-blue.svg)](https://pypi.org/project/pip/)
[![Neo4j Desktop](https://img.shields.io/badge/neo4jDesktop-1.4.10-blue.svg)](https://pypi.org/project/pip/)
![Neo4J 4.3.19](https://img.shields.io/badge/Neo4j-008CC1?style=for-the-badge&logo=neo4j&logoColor=white)

## Setup

[comment]: <> (### neo4j databse)

[comment]: <> (1. Import the database [neo4j-data.dump]&#40;neo4j-db/neo4j-data.dump&#41; in neo4j following these [instructions]&#40;https://tbgraph.wordpress.com/2020/11/11/dump-and-load-a-database-in-neo4j-desktop/comment-page-1/&#41;.)

## Workflow 

1. Download Neo4j Desktop (developed using version: 1.5.6)
2. Create neo4j database from dump [neo4j-all-data-mixed.dump](neo4j-all-data-mixed.dump) (developed using version 5.3.0)
   1. Add the authentication parameters in [neo4j_transactions.py](graph_processing/neo4j_transactions.py)
3. [Download](https://surfdrive.surf.nl/files/index.php/s/P5CIFS5wQWav7LR) test datasets
4. 

### Add new datasets 
1. Create datasets in [tfd_datasets](tfd_datasets)
   1. If the dataset is for classification, add it to [classification_datasets](tfd_datasets/classification_datasets.py)
   2. If the dataset if for regression, add it to [regression_datasets](tfd_datasets/regression_datasets.py)
   3. Ingest the datasets in neo4j

### python setup
1. Create virtual environment

`virtualenv {env-name}`

2. Activate environment 

`source {env-name}/bin/activate`

3. Install requirements 

`pip install -r requirements.txt`

## Run experiments
All the experiments are in [experiments/all_experiments.py](experiments/all_experiments.py).
Just run the `main` function and everything will start running. 
> Note: The experiments take a long time to run.

[comment]: <> (### Non-Aug baseline )

[comment]: <> (1. Locate the file [baseline.py]&#40;augmentation/baseline.py&#41;.)

[comment]: <> (2. Run the script. )

[comment]: <> (### JoinAll baseline.)

[comment]: <> (1. Locate the file [join-all.py]&#40;augmentation/join-all.py&#41;.)

[comment]: <> (2. Run the script. )

[comment]: <> (### BestRank approach. )

[comment]: <> (1. Locate the file [algorithm_pipeline.py]&#40;augmentation/algorithm_pipeline.py&#41;.)

[comment]: <> (2. Uncomment line 57: )

[comment]: <> (`pipeline&#40;datasets, k=1&#41;  # BestRank` )

[comment]: <> (3. Run the script. )

[comment]: <> (### Get top-k best ranked join paths)

[comment]: <> (1. Locate the file [algorithm_pipeline.py]&#40;augmentation/algorithm_pipeline.py&#41;.)

[comment]: <> (2. Uncomment line 58: )

[comment]: <> (`pipeline&#40;datasets&#41;  # Top-k` )

[comment]: <> (3. Run the script. )

## Datasets 

[other-data](other-data) folder contains the experimental data. 
1. [data](other-data/data) contains the real datasets collected from 

_Motl, Jan, and Oliver Schulte. "The CTU prague relational learning repository." arXiv preprint arXiv:1511.03086 (2015)._

   - [CiteSeer](https://relational.fit.cvut.cz/dataset/CiteSeer)
   - [CORA](https://relational.fit.cvut.cz/dataset/CORA)
   - [PubMed_Diabetes](https://relational.fit.cvut.cz/dataset/PubMed_Diabetes)
   - [WebKP](https://relational.fit.cvut.cz/dataset/WebKP)

2. [original](other-data/original) contains the original source of the fabricated datasets:
   * [titanic](https://www.kaggle.com/dmilla/introduction-to-decision-trees-titanic-dataset/data)
   * [kidney-disease](https://www.kaggle.com/akshayksingh/kidney-disease-dataset)
   * [steel-plate-fault](https://www.kaggle.com/bpkapkar/steel-plates-faults-detection?select=Variable+Descriptor.txt)
   * [football](https://www.kaggle.com/estefanytorres/international-football-matches-with-stats-201017?select=FutbolMatches.csv)

3. [decision-trees-split](other-data/decision-trees-split) contains fabricated data based on the above sources.

## Code structure
1. [arda](arda) module contains our implementation of the feature selection part of ARDA system as per
Algorithms 1, 2, 3 from the following paper:

_Chepurko, Nadiia, et al. "ARDA: Automatic Relational Data Augmentation for Machine Learning." Proceedings of the VLDB Endowment 13.9._

2. [augmentation](augmentation) module contains:
- The ranking function
- The data preparation pipeline
- All the decision trees algorithms used for experiments. The ID3 implementation was sourced from this [repository](https://github.com/arriadevoe/lambda-computer-science/blob/master/Unit-4-Build-Week-1/Gad_Decision_Tree_Classifier_Final.ipynb).

3. [data_preparation](data_preparation) module contains:
- Functions to ingest the data into neo4j 
- Pruning strategy 

4. [feature_selection](feature_selection) module contains:
- The feature selection methods tested and used in our approach.
- Util functions to compute different scores.

5. [utils](utils) module contains diverse functions used throughout the approach.

6. [Visualisations](Visualisations.ipynb) Jupyter Notebook is used to create the plots from the paper using the results from
[experiment_results](experiment_results). 


### Old unused code
The following modules are not used for this version of the code:
1. [augmentation_workshop](augmentation_workshop) - data structure and ranking function used for the workshop version

2. [classification_approach](classification_approach) - Old approach using a regressor to predict the ranking
3. [neo4j-db](neo4j-db) - Contains the dump of the neo4j database we used for the experiments. 
