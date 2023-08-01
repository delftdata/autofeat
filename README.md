# AutoFeat: Transitive Feature Discovery over Join Paths
This repo contains the development and experimental codebase of AutoFeat.


[![Python 3.7+](https://img.shields.io/badge/python-3.8.2-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![pip](https://img.shields.io/badge/pip-20.0.2-blue.svg)](https://pypi.org/project/pip/)
[![Neo4j Desktop](https://img.shields.io/badge/neo4jDesktop-1.4.10-blue.svg)](https://pypi.org/project/pip/)


# 1. Development 

## Local development

### Requirements
- Python 3.8
- Java (for data discovery only - [Valentine](https://github.com/delftdata/valentine))
- neo4j 5.1.0 or 5.3.0

### Python setup 

1. Create virtual environment

`python -m venv {env-name}`

2. Activate environment 

`source {env-name}/bin/activate`

3. Install requirements 

`pip install -e .`

#### Fix libomp
LighGBM on AutoGluon [gives Segmentation Fault](https://github.com/autogluon/autogluon/issues/1442) or won't run unless you install the corret libomp 
as described [here](https://github.com/autogluon/autogluon/pull/1453/files). 
Steps: 
```
wget https://raw.githubusercontent.com/Homebrew/homebrew-core/fb8323f2b170bd4ae97e1bac9bf3e2983af3fdb0/Formula/libomp.rb
brew uninstall libomp
brew install libomp.rb
rm libomp.rb
```


## Docker
The Docker image already contains all the necesarry for development.

1. Build necessary Docker containers (Note: This step takes a while)
``` bash
   docker-compose up -d --build
```


# 2. Data setup
[Download](https://surfdrive.surf.nl/files/index.php/s/vdlZIT70hZuoO8f) our experimental datasets and put them in [data/benchmark](data/benchmark).

To evaluate AutoFeat, we have two data settings: [benchmark setting](#benchmark-setting) and [data lake setting](#data-lake-setting). 

## Benchmark setting
1. Go to [config.py](src/feature_discovery/config.py) and set `DATASET_TYPE = "benchmark"
`
3. Create database `benchmark` in neo4j: 
```
create database benchmark
:use benchmark
```
4. Ingest data
```
feature-discovery-cli ingest-kfk-data
```


## Data Lake setting
1. Go to [config.py](src/feature_discovery/config.py) and set `NEO4J_DATABASE = 'lake'`
2. Create database `lake` in neo4j:
```
create database lake
:use lake
```  
3. Ingest data - depending on how many cores you have, this step can take up to 1-2h.
```
feature-discovery-cli ingest-data --data-discovery-threshold=0.55 --discover-connections-data-lake
```

## Ingest data in Docker

1. Bash into container 
```bash
   docker exec -it feature-discovery-runner /bin/bash
```
2. Ingest data in the database following the steps from your preferred scenario ([benchmark setting](#benchmark-setting) 
or [data lake setting](#data-lake-setting)).
   1. You can acess neo4j UI at [localhost:7474](localhost:7474) (no credentials needed). 




# Experiments

To run the experiments in Docker, first bash into the container: 
```bash
   docker exec -it feature-discovery-runner /bin/bash
```

## Run AutoFeat
`feature-discovery-cli --help` will show the commands for running experiments: 

1. `run-all` Runs all experiments (ARDA + base + AutoFeat).

` feature-discovery-cli run-all --help ` will show you the parameters needed for running 

2. `run-arda` Runs the ARDA experiments

` feature-discovery-cli run-arda --help ` will show you the parameters needed for running 

`--dataset-labels` has to be the label of one of the datasets from `datasets.csv` file which resides in [data/benchmark](data/benchmark).

`--results-file` by default the experiments are saved as CSV with a predefined filename in [results](/results)

Example:

`feature-discovery-cli run-arda --dataset-labels steel` Will run the experiments on the _steel_ dataset and the results 
are saved in [results folder](results)


3. `run-base` Runs the base experiments

` feature-discovery-cli run-base --help ` will show you the parameters needed for running 

`--dataset-labels` has to be the label of one of the datasets from `datasets.csv` file which resides in [data/benchmark](data/benchmark).

`--results-file` by default the experiments are saved as CSV with a predefined filename.

Example: 

`feature-discovery-cli run-base --dataset-labels steel` Will run the experiments on the _steel_ dataset and the results 
are saved in [results folder](results)

4. `run-tfd` Runs the AutoFeat experiments.   

` feature-discovery-cli run-tfd --help ` will show you the parameters needed for running 

`--dataset-labels` has to be the label of one of the datasets from `datasets.csv` file which resides in [data/benchmark](data/benchmark).

`--results-file` by default the experiments are saved as CSV with a predefined filename.

`--value-ratio` one of the hyper-parameters of our approach, it represents a data quality metric - the percentage of 
null values allowed in the datasets. Default: 0.55

`--top-k` one of the hyper-parameters of our approach, 
it represents the number of features to select from each dataset and the number of paths. Default: 15 

Example: 

`feature-discovery-cli run-tfd --dataset-labels steel` Will run the experiments on the _steel_ 
dataset and the results are saved in [results folder](results)

## Datasets 

Main [source](https://huggingface.co/datasets/inria-soda/tabular-benchmark#source-data) for finding datasets.

| Dataset Label | Source | Processing strategy | 
| ------------- | ------ | --------- | 
| [jannis](data/jannis) | [openml](https://www.openml.org/search?type=data&sort=runs&id=45021&status=active) | [short_reverse_correlation](https://github.com/kirilvasilev16/PythonTableDivider) | 
| [MiniBooNe](data/miniboone) | [openml](https://www.openml.org/search?type=data&sort=runs&id=44128&status=active) | [short_reverse_correlation](https://github.com/kirilvasilev16/PythonTableDivider) | 
| [covertype](data/covertype) | [openml](https://www.openml.org/search?type=data&sort=runs&id=44159&status=active) | [short_reverse_correlation](https://github.com/kirilvasilev16/PythonTableDivider) | 
| [EyeMovement](data/eyemove) | [openml](https://www.openml.org/search?type=data&sort=runs&id=44157&status=active) | [short_reverse_correlation](https://github.com/kirilvasilev16/PythonTableDivider) |
| [Bioresponse](data/bioresponse) | [openml](https://www.openml.org/search?type=data&sort=runs&id=45019&status=active) | [short_reverse_correlation](https://github.com/kirilvasilev16/PythonTableDivider) | 
| [school](data/school) | [ARDA Paper](http://www.vldb.org/pvldb/vol13/p1373-chepurko.pdf) | None | 
| [steel](data/steel) | [openml](https://www.openml.org/search?type=data&sort=runs&status=active&qualities.NumberOfClasses=%3D_2&id=1504) | [short_reverse_correlation](https://github.com/kirilvasilev16/PythonTableDivider) |
| [credit](data/credit) | [openml](https://www.openml.org/search?type=data&sort=runs&status=active&qualities.NumberOfClasses=%3D_2&id=31) | [short_reverse_correlation](https://github.com/kirilvasilev16/PythonTableDivider) |

## Plots

1. To recreate our plots, first download the results from [here](https://surfdrive.surf.nl/files/index.php/s/fIhQNikpFbemozv).
 
2. Then, open the jupyter notebook. Run in the root folder of the project: 
```bash
jupyter notebook
```

2. Open the file [Visualisations.ipynb](Visualisations.ipynb).
3. Run every cell. 


### Maintainer
Created and maintained by [Andra Ionescu](https://andraionescu.github.io)