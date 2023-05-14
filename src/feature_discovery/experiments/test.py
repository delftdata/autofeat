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


def test_base_accuracy(dataset: Dataset, autogluon: bool = True):
    print(f"Base result on table {dataset.base_table_id}")
    dataframe = pd.read_csv(dataset.base_table_id, header=0, engine="python", encoding="utf8", quotechar='"',
                            escapechar='\\')

    if autogluon:
        best_result, all_results = run_auto_gluon(approach=Result.BASE,
                                                  dataframe=dataframe,
                                                  target_column=dataset.target_column,
                                                  data_label=dataset.base_table_label,
                                                  join_name=dataset.base_table_label,
                                                  algorithms_to_run=hyper_parameters)
        return all_results
    else:
        entry = train_test_cart(train_data=dataframe,
                                target_column=dataset.target_column,
                                regression=dataset.dataset_type)
        entry.approach = Result.BASE
        entry.data_label = dataset.base_table_label
        entry.data_path = dataset.base_table_label

        return [entry]


def test_arda(dataset: Dataset, sample_size: int = 1000, autogluon: bool = True) -> List:
    from arda.arda import select_arda_features_budget_join

    print(f"ARDA result on table {dataset.base_table_id}")

    start = time.time()
    dataframe, dataframe_label, selected_features, join_name = select_arda_features_budget_join(
        base_node_id=str(dataset.base_table_id),
        target_column=dataset.target_column,
        sample_size=sample_size,
        regression=dataset.dataset_type)
    end = time.time()
    print(f"X shape: {dataframe.shape}\nSelected features:\n\t{selected_features}")

    if len(selected_features) == 0:
        from algorithms import CART
        entry = Result(
            algorithm=CART.LABEL,
            accuracy=0,
            feature_importance={},
            feature_selection_time=end - start,
            approach=Result.ARDA,
            data_label=dataset.base_table_label,
            data_path=join_name,
            join_path_features=selected_features
        )
        entry.total_time += entry.feature_selection_time
        return [entry]

    features = selected_features.copy()
    features.append(dataset.target_column)

    if autogluon:
        best_result, all_results = run_auto_gluon(approach=Result.ARDA,
                                                  dataframe=dataframe[features],
                                                  target_column=dataset.target_column,
                                                  data_label=dataset.base_table_label,
                                                  join_name=join_name,
                                                  algorithms_to_run=hyper_parameters)
        for result in all_results:
            result.feature_selection_time = end - start
            result.total_time += result.feature_selection_time
        return all_results
    else:
        entry = train_test_cart(train_data=dataframe[features],
                                target_column=dataset.target_column,
                                regression=dataset.dataset_type)
        entry.feature_selection_time = end - start
        entry.total_time += entry.feature_selection_time
        entry.approach = Result.ARDA
        entry.data_label = dataset.base_table_label
        entry.data_path = join_name
        entry.join_path_features = selected_features

        print(entry)

        return [entry]


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


def test_bfs_pipeline(dataset: Dataset, value_ratio: float = 0.55, gini: bool = False) -> List:
    print(f"BFS result with table {dataset.base_table_id}")

    start = time.time()
    bfs_traversal = BfsAugmentation(base_table_label=dataset.base_table_label,
                                    target_column=dataset.target_column,
                                    value_ratio=value_ratio,
                                    auto_gluon=True)
    bfs_traversal.bfs_traverse_join_pipeline(queue={str(dataset.base_table_id)})
    end = time.time()

    print("FINISHED BFS")

    # Aggregate results
    results = []
    for join_name in bfs_traversal.join_name_mapping.keys():
        aux = list(filter(lambda x: x.data_path == join_name, bfs_traversal.all_results))

        for item in aux:
            item.feature_selection_time = end - start
            item.total_time += item.feature_selection_time
            results.append(item)

    # Save results
    pd.DataFrame(results).to_csv(
        RESULTS_FOLDER / f"results_{dataset.base_table_label}_bfs_{value_ratio}_autogluon_all_mixed.csv", index=False)
    pd.DataFrame.from_dict(bfs_traversal.join_name_mapping, orient='index', columns=["join_name"]).to_csv(
        RESULTS_FOLDER / f'join_mapping_{dataset.base_table_label}_bfs_{value_ratio}_autogluon_all_mixed.csv')

    return results


def ablation_study_enumerate_paths(datasets: List[Dataset], value_ratio: float):
    results = {"study": "enumerate"}
    for dataset in datasets:
        bfs_traversal = BfsAugmentation(base_table_label=dataset.base_table_label,
                                        target_column=dataset.target_column,
                                        value_ratio=value_ratio)
        total_time = bfs_traversal.enumerate_all_paths(queue={str(dataset.base_table_id)})
        results[f"{dataset.base_table_label}_paths"] = len(bfs_traversal.total_paths.keys())
        results[f"{dataset.base_table_label}_features"] = bfs_traversal.total_paths
        results[f"{dataset.base_table_label}_runtime"] = total_time

    return results


def ablation_study_prune_paths(datasets: List[Dataset], value_ratio: float):
    results = {"study": "enumerate_prune"}
    for dataset in datasets:
        bfs_traversal = BfsAugmentation(base_table_label=dataset.base_table_label,
                                        target_column=dataset.target_column,
                                        value_ratio=value_ratio)
        total_time = bfs_traversal.prune_paths(queue={str(dataset.base_table_id)})
        results[f"{dataset.base_table_label}_paths"] = len(bfs_traversal.total_paths.keys())
        results[f"{dataset.base_table_label}_features"] = bfs_traversal.total_paths
        results[f"{dataset.base_table_label}_runtime"] = total_time

    return results


def ablation_study_enumerate_and_join(datasets: List[Dataset], value_ratio: float):
    results = {"study": "enumerate_join"}
    for dataset in datasets:
        bfs_traversal = BfsAugmentation(base_table_label=dataset.base_table_label,
                                        target_column=dataset.target_column,
                                        value_ratio=value_ratio)
        total_time = bfs_traversal.enumerate_and_join(queue={str(dataset.base_table_id)})
        results[f"{dataset.base_table_label}_paths"] = len(bfs_traversal.total_paths.keys())
        results[f"{dataset.base_table_label}_features"] = bfs_traversal.total_paths
        results[f"{dataset.base_table_label}_runtime"] = total_time
    return results


def ablation_study_feature_selection(datasets: List[Dataset], value_ratio: float):
    results = {"study": "enumerate_join_prune_fs"}
    for dataset in datasets:
        bfs_traversal = BfsAugmentation(base_table_label=dataset.base_table_label,
                                        target_column=dataset.target_column,
                                        value_ratio=value_ratio)
        total_time = bfs_traversal.apply_feature_selection(queue={str(dataset.base_table_id)})
        results[f"{dataset.base_table_label}_paths"] = len(bfs_traversal.total_paths.keys())
        results[f"{dataset.base_table_label}_features"] = bfs_traversal.total_paths
        results[f"{dataset.base_table_label}_runtime"] = total_time
    return results


def ablation_study_prune_join_key_level(datasets: List[Dataset], value_ratio: float):
    results = {"study": "enumerate_join_prune_fs_rank_jk"}
    for dataset in datasets:
        bfs_traversal = BfsAugmentation(base_table_label=dataset.base_table_label,
                                        target_column=dataset.target_column,
                                        value_ratio=value_ratio)
        total_time = bfs_traversal.prune_join_key_level(queue={str(dataset.base_table_id)})
        results[f"{dataset.base_table_label}_paths"] = len(bfs_traversal.total_paths.keys())
        results[f"{dataset.base_table_label}_features"] = bfs_traversal.total_paths
        results[f"{dataset.base_table_label}_runtime"] = total_time
    return results


def all_ablation(datasets: List[Dataset], value_ratio: float):
    all_results = []

    result = ablation_study_enumerate_paths(datasets, value_ratio=value_ratio)
    all_results.append(result)

    result = ablation_study_enumerate_and_join(datasets, value_ratio=value_ratio)
    all_results.append(result)

    result = ablation_study_prune_paths(datasets, value_ratio=value_ratio)
    all_results.append(result)

    result = ablation_study_feature_selection(datasets, value_ratio=value_ratio)
    all_results.append(result)

    result = ablation_study_prune_join_key_level(datasets, value_ratio=value_ratio)
    all_results.append(result)

    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / f'ablation_study_{value_ratio}_autogluon.csv', index=False)


def tune_value_ratio_threshold(datasets: List[Dataset]):
    all_results = []
    value_ratio_threshold = np.arange(0.15, 1.05, 0.05)
    for threshold in value_ratio_threshold:
        for dataset in datasets:
            result_bfs = test_bfs_pipeline(dataset, value_ratio=threshold)
            all_results.extend(result_bfs)
    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / f"all_results_value_ratio_tuning_2.csv", index=False)


def aggregate_results():
    all_results = []

    for dataset in CLASSIFICATION_DATASETS:
        # for dataset in REGRESSION_DATASETS:
        # result_bfs = test_bfs_pipeline(dataset, value_ratio=0.65)
        # all_results.extend(result_bfs)
        # result_base = test_base_accuracy(dataset)
        # all_results.extend(result_base)
        result_arda = test_arda(dataset, sample_size=3000)
        all_results.extend(result_arda)
        # result_dfs = test_dfs_pipeline(dataset, value_ratio=0.5)
        # all_results.extend(result_dfs)

    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / f"all_results_autogluon_ARDA.csv", index=False)


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

# test_bfs_pipeline(school_small, value_ratio=0.65)
# test_dfs_pipeline()
# test_base_accuracy(accounting)
# test_arda(credit, sample_size=3000)
# aggregate_results()

ablation_study_enumerate_paths([school_small], value_ratio=0.65)
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

# tune_value_ratio_threshold(CLASSIFICATION_DATASETS)

# all_ablation(CLASSIFICATION_DATASETS, value_ratio=0.65)
