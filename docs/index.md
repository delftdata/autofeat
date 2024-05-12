## Description

***AutoFeat*** is an open-source automatic approach for feature discovery on tabular datasets. 

Given a base table with a target variable and a repository of tabular datasets, AutoFeat helps to discover relevant 
features for augmentation among the tables from the data repository. The resulting augmented table will be a better 
training dataset for decision tree Machine Learning (ML) algorithms. 


## Authors

<figure class="item author-image">
    <a href="https://andraionescu.github.io/"><img src="./assets/img/andra_ionescu.jpg" height="auto" width="80" style="border-radius:50%"/></a>
    <figcaption class="caption" style="display:block">Andra Ionescu <br>TU Delft</figcaption>
</figure>

<figure class="item author-image">
    <a href="https://www.linkedin.com/in/kiril-vasilev/"><img src="./assets/img/kiril_vasilev.jpeg" height="auto" width="80" style="border-radius:50%"/></a>
    <figcaption class="caption" style="display:block">Kiril Vasilev <br>TU Delft</figcaption>
</figure>


<figure class="item author-image">
    <a href="https://www.linkedin.com/in/florenabuse/"><img src="./assets/img/florena_buse.jpeg" height="auto" width="80" style="border-radius:50%"/></a>
    <figcaption class="caption" style="display:block">Florena Buse <br> TU Delft</figcaption>
</figure>


<figure class="item author-image">
    <a href="https://rihanhai.com/"><img src="./assets/img/rihan_hai.jpg" height="auto" width="80" style="border-radius:50%"/></a>
    <figcaption class="caption" style="display:block">Rihan Hai <br> TU Delft</figcaption>
</figure>


<figure class="item author-image">
    <a href="http://asterios.katsifodimos.com/"><img src="./assets/img/asterios_katsifodimos.jpg" height="auto" width="80" style="border-radius:50%"/></a>
    <figcaption class="caption" style="display:block">Asterios Katsifodimos <br>TU Delft</figcaption>
</figure>

## AutoFeat Methods

- Dataset Discovery: AutoFeat uses [Valentine](https://delftdata.github.io/valentine/) to discover joinable tables.
  - Graph Traversal: AutoFeat uses Breadth First Search to traverse the graph of connections, which helps us manage the error propagation. 
- Streaming Feature Selection: AutoFeat uses streaming feature selection to navigate the space of joinable tables and select the relevant features for augmentation. 
  - Relevance: AutoFeat measures the relevance of features using Pearson correlation.
  - Redundancy: AutoFeat removes redundant features using Minimum Redundancy Maximum Relevance algorithm.

## Datasets


| Dataset Source                                                                                                   | # Rows |    Processing strategy    | # Joinable Tables | # Total Features |                                                                  Links                                                                  |
|------------------------------------------------------------------------------------------------------------------|:------:|:-------------------------:|:-----------------:|:----------------:|:---------------------------------------------------------------------------------------------------------------------------------------:|
| [jannis](https://www.openml.org/search?type=data&sort=runs&id=45021&status=active)                               | 57581  | short_reverse_correlation |        12         |        55        |                              [processed data](https://surfdrive.surf.nl/files/index.php/s/OdYbGOwWHytkBdE)                              |
| [miniboone](https://www.openml.org/search?type=data&sort=runs&id=44128&status=active)                            | 73000  | short_reverse_correlation |        15         |        51        |                              [processed data](https://surfdrive.surf.nl/files/index.php/s/OdYbGOwWHytkBdE)                              |
| [covertype](https://www.openml.org/search?type=data&sort=runs&id=44159&status=active)                            | 423682 | short_reverse_correlation |        12         |        21        |                              [processed data](https://surfdrive.surf.nl/files/index.php/s/OdYbGOwWHytkBdE)                              |
| [eyemove](https://www.openml.org/search?type=data&sort=runs&id=44157&status=active)                              |  7609  | short_reverse_correlation |         6         |        24        |                              [processed data](https://surfdrive.surf.nl/files/index.php/s/OdYbGOwWHytkBdE)                              |
| [credit](https://www.openml.org/search?type=data&sort=runs&status=any&id=31)                                     |  1001  | short_reverse_correlation |         5         |        21        |                              [processed data](https://surfdrive.surf.nl/files/index.php/s/OdYbGOwWHytkBdE)                              |
| [bioresponse](https://www.openml.org/search?type=data&sort=runs&id=45019&status=active)                          |  3435  | short_reverse_correlation |        40         |       420        |                              [procssed data](https://surfdrive.surf.nl/files/index.php/s/OdYbGOwWHytkBdE)                               | 
| [steel](https://www.openml.org/search?type=data&sort=runs&status=active&qualities.NumberOfClasses=%3D_2&id=1504) |  1943  | short_reverse_correlation |        15         |        34        |                              [processed data](https://surfdrive.surf.nl/files/index.php/s/OdYbGOwWHytkBdE)                              |
| [school](https://arxiv.org/pdf/2003.09758)                                                                       |  1775  |           None            |        16         |       731        |                             [original data](https://surfdrive.surf.nl/files/index.php/s/9zye8gWOvc96iWY)                                |


## Repository

- <https://github.com/delftdata/autofeat> : Main repository containing the AutoFeat source code. 
- <https://github.com/kirilvasilev16/PythonTableDivider> : Repository containing the dataset processing strategies. 
- <https://github.com/delftdata/bsc_research_project_q4_2023/tree/main/autofeat_experimental_analysis> : Repository containing the evaluation of relevance and redundancy methods.  

## AutoFeat Papers
- [[Pre-print](assets/papers/ICDE_FeatureDiscovery.pdf)] AutoFeat: Transitive Feature Discovery over Join Paths 

## ICDE 2024 

- ICDE 2024 Poster 

[Poster - companion to the paper](assets/poster/AutoFeat_poster-no-bleed.pdf)


- ICDE 2024 Presentation

[![Slides](assets/presentation/slide_0.png)](assets/presentation/AutoFeat-presentation.pdf)

