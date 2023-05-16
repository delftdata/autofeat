# Transitive Feature Discovery over Join Paths

[![Python 3.7+](https://img.shields.io/badge/python-3.8.2-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![pip](https://img.shields.io/badge/pip-20.0.2-blue.svg)](https://pypi.org/project/pip/)
[![Neo4j Desktop](https://img.shields.io/badge/neo4jDesktop-1.4.10-blue.svg)](https://pypi.org/project/pip/)
![Neo4J 4.3.19](https://img.shields.io/badge/Neo4j-008CC1?style=for-the-badge&logo=neo4j&logoColor=white)


# Run With Docker
1. Build necessary Docker containers
``` bash
   docker-compose up -d --build
```

2. Load Data into Neo4J DB 
``` bash
   docker exec -it feature-discovery-neo4j /bin/bash -c "/neo4j-db/neo4j-entrypoint.sh"
```

3. Bash into container and run experiments
``` bash
   docker exec -it feature-discovery-runner /bin/bash
```
## Setup

### python setup
1. Create virtual environment

`virtualenv {env-name}`

2. Activate environment 

`source {env-name}/bin/activate`

3. Install requirements 

`pip install -e .`

### Fix libomp
LighGBM on AutoGluon [gives Segmentation Fault](https://github.com/autogluon/autogluon/issues/1442) or won't run unless you install the corret libomp 
as described [here](https://github.com/autogluon/autogluon/pull/1453/files). 
Steps: 
```
wget https://raw.githubusercontent.com/Homebrew/homebrew-core/fb8323f2b170bd4ae97e1bac9bf3e2983af3fdb0/Formula/libomp.rb
brew uninstall libomp
brew install libomp.rb
rm libomp.rb
```


[comment]: <> (### neo4j databse)

[comment]: <> (1. Import the database [neo4j-data.dump]&#40;neo4j-db/neo4j-data.dump&#41; in neo4j following these [instructions]&#40;https://tbgraph.wordpress.com/2020/11/11/dump-and-load-a-database-in-neo4j-desktop/comment-page-1/&#41;.)

## Workflow 

### Work with our test datasets

1. Download Neo4j Desktop (developed using version: 1.5.6).
2. Create neo4j database from dump [neo4j-all-data-mixed.dump](neo4j-all-data-mixed.dump) (developed using version 5.3.0).
   1. Add the authentication parameters in [config](src/feature_discovery/config.py).
3. [Download](https://surfdrive.surf.nl/files/index.php/s/P5CIFS5wQWav7LR) test datasets.

### (or) add new datasets 
1. Create a folder <folder_name> in [data](data).
2. Add your data in <folder_name> folder.
3. Add a new line in [datasets](data/datasets.csv) to identify the new dataset. 

Example: 

| base_table_path | base_table_name | base_table_label | target_column | dataset_type |
| --------------- | --------------- | ---------------- | ------------- | ------------ |
| "school" | "base.csv" | "school" | "class" | classification |

`base_table_path` is the <folder_name> where you added the data.

`base_table_name` is the name of the table that you want to augment with new features.

`base_table_label` string used to identify your dataset (can be the same as <folder_name> if <folder_name> is human 
readable).

`target_column` the target/label feature containing the class labels. 

`dataset_type` - "classification" if the dataset is used for classification or "regression" if the dataset
if used for regression problems. 

4. Ingest the new dataset 
`feature-discovery-cli ingest-data --dataset_label <base_table_label>`

> Note: `--discover_connections_data_lake` will create even more connections and simulate a real life data lake scenario.
> However, running the experiments with `--discover_connections_data_lake` flag increases the runtime exponentially. 
> 
> Best: Let it run overnight when using `--discover_connections_data_lake`


### Run experiments
`feature-discovery-cli --help` will show the commands for running experiments: 

1. `run-all` Runs all experiments (ARDA + base + TFD).
` feature-discovery-cli run-all --help ` will show you the parameters needed for running 
2. `run-arda` Runs the ARDA experiments
` feature-discovery-cli run-arda --help ` will show you the parameters needed for running 

`--dataset-labels` has to be the label of one of the datasets from [tfd_datasets](src/feature_discovery/tfd_datasets)

`--results-file` by default the experiments are saved as CSV with a predefined filename.

Example:

`feature-discovery-cli run-arda --dataset-labels steel` Will run the experiments on the _steel_ dataset and the results 
are saved in [results folder](results)


3. `run-base` Runs the base experiments
` feature-discovery-cli run-base --help ` will show you the parameters needed for running 

`--dataset-labels` has to be the label of one of the datasets from [tfd_datasets](src/feature_discovery/tfd_datasets)

`--results-file` by default the experiments are saved as CSV with a predefined filename.

Example: 

`feature-discovery-cli run-base --dataset-labels steel` Will run the experiments on the _steel_ dataset and the results 
are saved in [results folder](results)

4. `run-tfd` Runs the TFD experiments.   
` feature-discovery-cli run-tfd --help ` will show you the parameters needed for running 

`--dataset-labels` has to be the label of one of the datasets from [tfd_datasets](src/feature_discovery/tfd_datasets)

`--results-file` by default the experiments are saved as CSV with a predefined filename.

`--value-ratio` one of the hyper-parameters of our approach, it represents a data quality metric - the percentage of 
null values (1-value_ratio) allowed in the datasets. Default: 0.55

`--auto-gluon` Runs the experiments using AutoGluon framework. Default True. 

Example: 

`feature-discovery-cli run-tfd --dataset-labels steel --value-ratio 0.65` Will run the experiments on the _steel_ 
dataset and the results are saved in [results folder](results)


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

