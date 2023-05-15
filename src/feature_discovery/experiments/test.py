import json
import time
from typing import List

import numpy as np
import pandas as pd

from feature_discovery.augmentation.bfs_pipeline import BfsAugmentation
from feature_discovery.augmentation.trial_error import dfs_traverse_join_pipeline, train_test_cart, run_auto_gluon
from feature_discovery.config import RESULTS_FOLDER, JOIN_RESULT_FOLDER, ROOT_FOLDER
from feature_discovery.data_preparation.dataset_base import Dataset
from feature_discovery.experiments.result_object import Result
from feature_discovery.graph_processing.traverse_graph import dfs_traversal
from feature_discovery.tfd_datasets import CLASSIFICATION_DATASETS, school_small

hyper_parameters = {'RF': {}, 'GBM': {}, 'XGB': {}, 'XT': {}}


def test_dfs_pipeline(dataset: Dataset, value_ratio: float = 0.55, gini: bool = False) -> List:
    print(f"DFS result with table {dataset.base_table_id}")

    start = time.time()
    join_path_tree = {}
    dfs_traversal(base_node_id=str(dataset.base_table_id), discovered=[], join_tree=join_path_tree)

    join_name_mapping = {}
    paths_score = {}
    train_results = []
    all_paths = dfs_traverse_join_pipeline(base_node_id=str(dataset.base_table_id), target_column=dataset.target_column,
                                           base_table_label=dataset.base_table_label, join_tree=join_path_tree,
                                           join_name_mapping=join_name_mapping, value_ratio=value_ratio,
                                           paths_score=paths_score, gini=gini)
    end = time.time()
    print(f"FINISHED DFS")

    # Train, test each path
    print(f"TRAIN WITHOUT feature selection")
    for join_name in join_name_mapping.keys():
        joined_df = pd.read_csv(JOIN_RESULT_FOLDER / dataset.base_table_label / join_name_mapping[join_name], header=0,
                                engine="python", encoding="utf8", quotechar='"', escapechar='\\')
        result = train_test_cart(dataframe=joined_df,
                                 target_column=dataset.target_column,
                                 regression=dataset.dataset_type)
        result.feature_selection_time = end - start
        result.approach = "TFD_DFS"
        result.data_label = dataset.base_table_label
        result.data_path = join_name
        train_results.append(result)

    # Save results
    pd.Series(list(all_paths), name="filename").to_csv(
        RESULTS_FOLDER / f"paths_{dataset.base_table_label}_dfs_{value_ratio}.csv", index=False)
    pd.DataFrame(train_results).to_csv(
        RESULTS_FOLDER / f"results_{dataset.base_table_label}_dfs_{value_ratio}.csv", index=False)
    pd.DataFrame.from_dict(join_name_mapping, orient='index', columns=["join_name"]).to_csv(
        RESULTS_FOLDER / f'join_mapping_{dataset.base_table_label}_dfs_{value_ratio}.csv')
    with open(RESULTS_FOLDER / f"all_paths_{dataset.base_table_label}_dfs_{value_ratio}.json", "w") as f:
        json.dump(paths_score, f)
    with open(f'join_tree_{dataset.base_table_label}_dfs.json', 'w') as f:
        json.dump(join_path_tree, f)

    return train_results


def test_autogluon():
    from autogluon.tabular import TabularDataset, TabularPredictor
    from autogluon.features.generators import AutoMLPipelineFeatureGenerator
    from sklearn.model_selection import train_test_split

    train_data = TabularDataset(f'{ROOT_FOLDER}/other-data/original/titanic-og.csv')
    label = 'Survived'

    print(train_data.head())
    auto_ml_pipeline_feature_generator = AutoMLPipelineFeatureGenerator(
        enable_text_special_features=False,
        enable_text_ngram_features=False)
    tf_data = auto_ml_pipeline_feature_generator.fit_transform(X=train_data)
    print(tf_data.head())

    X_train, X_test, y_train, y_test = train_test_split(train_data.drop(columns=[label]), train_data[label],
                                                        test_size=0.2,
                                                        random_state=10)
    train = X_train.copy()
    train[label] = y_train

    test = X_test.copy()
    test[label] = y_test

    exclude_models = ['NN_TORCH', 'FASTAI', 'AG_AUTOMM', 'FT_TRANSFORMER', 'FASTTEXT', 'VW', 'AG_TEXT_NN',
                      'AG_IMAGE_NN', 'WeightedEnsemble_L2']

    predictor = TabularPredictor(label=label,
                                 problem_type="binary",
                                 verbosity=2).fit(train_data=train,
                                                  hyperparameters={'RF': {}, 'GBM': {}, 'XGB': {}, 'XT': {}})
    # predictor.evaluate()
    res = []
    all_results = predictor.info()
    for model in all_results["model_info"].keys():
        # values = predictor.evaluate(test, model=model)
        ft_imp = predictor.feature_importance(data=test, model=model, feature_stage='original')
        print(ft_imp)
        entry = Result(
            algorithm=model,
            accuracy=all_results["model_info"][model]['val_score'],
            feature_importance=dict(zip(list(ft_imp.index), ft_imp['importance'])),
            train_time=all_results["model_info"][model]['fit_time']
        )
        res.append(entry)

    print(res)

    # result = predictor.evaluate(test, model="LightGBM")
    # print(result)
    # ft_imp = predictor.feature_importance(data=test, model="LightGBM", feature_stage='transformed_model')
    # print(ft_imp)
    # features = predictor.info()
    # print(features)
    # leaderboard = predictor.leaderboard()


# test_autogluon()

# ablation_study_enumerate_paths([school_small], value_ratio=0.65)
# {'nyc': (11, 16.809264183044434), 'school': (1092, 2.2020440101623535), 'credit': (1, 0.5188112258911133),
#  'steel': (183, 0.6745116710662842)}

# ablation_study_enumerate_and_join(CLASSIFICATION_DATASETS, value_ratio=0.45)
# {'nyc': (11, 95.02598476409912), 'school': (1019, 1121.9258248806), 'credit': (1, 2.9994161128997803),
#  'steel': (83, 36.911052942276)}

# ablation_study_prune_paths(CLASSIFICATION_DATASETS, value_ratio=0.45)
# {'nyc': (3, 45.107990026474), 'school': (52, 73.44366002082825), 'credit': (1, 2.2596540451049805),
#  'steel': (63, 25.328887224197388)}

# ablation_study_feature_selection(CLASSIFICATION_DATASETS, value_ratio=0.45)
# {'nyc': (3, 58.07925200462341), 'school': (31, 238.45129704475403), 'credit': (1, 2.773263931274414),
#  'steel': (1, 4.591271162033081)}

# ablation_study_prune_join_key_level(CLASSIFICATION_DATASETS, value_ratio=0.45)
# {'nyc': (4, 70.50428295135498), 'school': (35, 377.25908303260803), 'credit': (1, 3.509957790374756),
#  'steel': (37, 201.2652440071106)}
# {'nyc': (3, 65.19033694267273), 'school': (57, 928.0529820919037), 'credit': (1, 4.812485933303833),
#  'steel': (1, 5.542677164077759)}
