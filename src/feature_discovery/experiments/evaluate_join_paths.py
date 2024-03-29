import logging
from typing import Tuple, List

import pandas as pd
import tqdm

from feature_discovery.autofeat_pipeline.autofeat import AutoFeat
from feature_discovery.autofeat_pipeline.join_path_utils import get_path_length
from feature_discovery.config import RESULTS_FOLDER
from feature_discovery.experiments.evaluation_algorithms import evaluate_all_algorithms
from feature_discovery.experiments.init_datasets import ALL_DATASETS
from feature_discovery.experiments.result_object import Result
from feature_discovery.experiments.utils_dataset import filter_datasets
from feature_discovery.helpers.read_data import get_df_with_prefix


def evaluate_paths(bfs_result: AutoFeat, problem_type: str, algorithm: str, top_k_paths: int = 15) -> Tuple[List[Result], List[Tuple]]:
    logging.debug(f"Evaluate top-{top_k_paths} paths ... ")
    sorted_paths = sorted(bfs_result.ranking.items(), key=lambda r: (r[1], -get_path_length(r[0])), reverse=True)
    top_k_path_list = sorted_paths if len(sorted_paths) < top_k_paths else sorted_paths[:top_k_paths]
    base_features = bfs_result.partial_join_selected_features[bfs_result.base_table_id]

    all_results = []
    for path in tqdm.tqdm(top_k_path_list):
        join_name, rank = path
        if join_name == bfs_result.base_table_id:
            continue

        features = bfs_result.partial_join_selected_features[join_name]
        features.append(bfs_result.target_column)
        features.extend(base_features)
        logging.debug(f"Feature before join_key removal:\n{features}")
        # features = list((set(features) - set(bfs_result.join_keys[join_name])).intersection(set(dataframe.columns)))
        # logging.debug(f"Feature after join_key removal:\n{features}")

        features_tables = [f"{feat.split('.csv')[0]}.csv" for feat in features]
        features_tables.sort()
        features_tables = set(features_tables)

        path_tables = {}
        for p in join_name.split("--"):
            aux = p.split("-")
            if len(aux) == 4:
                path_tables[aux[3]] = (aux[0], aux[1], aux[2], aux[3])

        path_list = []
        for table in features_tables:
            if table in path_list:
                continue
            path_aux = create_join_tree(table, path_tables)
            if not (type(path_aux) is list) and (path_aux not in path_tables.keys()):
                continue
            path_list.append(path_aux)

        dataframe = join_from_path(path_list, bfs_result.target_column, bfs_result.base_table_id)
        features = list(set(features).intersection(set(dataframe.columns)))

        if len(features) < 2:
            features = bfs_result.partial_join_selected_features[bfs_result.base_table_id]
            features.append(bfs_result.target_column)

        results, _ = evaluate_all_algorithms(dataframe=dataframe[features],
                                             target_column=bfs_result.target_column,
                                             problem_type=problem_type,
                                             algorithm=algorithm)
        for result in results:
            result.rank = rank
            result.data_path = path_list
        all_results.extend(results)

        dataframe = None

    return all_results, top_k_path_list


def evaluate_paths_from_file(filename: str, algorithm: str, top_k_paths: int = 15) -> List[Result]:
    logging.debug(f"Evaluate top-{top_k_paths} paths ... ")

    data = pd.read_csv(RESULTS_FOLDER / filename)

    data_paths = data[~data['data_label'].isin(['air', 'yprop', 'superconduct'])]
    data_paths = data_paths.loc[data_paths.groupby(by=['data_path'])['accuracy'].idxmax()]

    all_results = []
    for index, row in data_paths.iterrows():
        dataset = filter_datasets([row['data_label']])[0]
        path_list = pd.eval(row['data_path'])
        rank = pd.eval(row['rank'])
        features = pd.eval(row['join_path_features'])
        logging.debug(f"Feature before join_key removal:\n{features}")

        dataframe = join_from_path(path_list, dataset.target_column, dataset.base_table_id)
        features = list(set(features).intersection(set(dataframe.columns)))
        target = f"{dataset.base_table_label}/{dataset.base_table_name}.{dataset.target_column}"
        features.append(target)

        results, _ = evaluate_all_algorithms(dataframe=dataframe[features],
                                             target_column=target,
                                             algorithm=algorithm)
        for result in results:
            result.rank = rank
            result.data_path = path_list
            result.approach = Result.TFD
            result.feature_selection_time = pd.eval(row['feature_selection_time'])
            result.total_time += pd.eval(row['total_time'])
            result.top_k = pd.eval(row['top_k'])
            result.data_label = dataset.base_table_label
            result.cutoff_threshold = pd.eval(row['cutoff_threshold'])
        all_results.extend(results)

        dataframe = None

    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / f"results_autofeat_from_path_{algorithm}.csv", index=False)

    return all_results


def join_from_path(path, target, base_node):
    join_path = ''
    step = 3
    joined_df = None
    for p in path:
        if ".".join(p) in join_path:
            continue
        for i, el in enumerate(p[::step]):
            if (i * step) + 1 == len(p):
                continue

            aux = ".".join([p[i], p[i + 1], p[i + 2], p[(i + 1) * step]])
            if aux in join_path:
                continue

            if p[i * step] not in join_path:
                if p[i * step] == base_node:
                    left_table, _ = get_df_with_prefix(p[i * step], target)
                else:
                    left_table, _ = get_df_with_prefix(p[i * step])
            else:
                left_table = joined_df

            right_table, _ = get_df_with_prefix(p[(i + 1) * step])
            to_column = f"{p[(i + 1) * step]}.{p[i * step + 2]}"
            right_table = right_table.groupby(to_column).sample(n=1, random_state=42)

            joined_df = pd.merge(
                left_table,
                right_table,
                how="left",
                left_on=f"{p[i * step]}.{p[i * step + 1]}",
                right_on=f"{p[(i + 1) * step]}.{p[i * step + 2]}",
            )

            join_path += aux

    return joined_df


def create_join_tree(table, path_tables):
    if table in path_tables.keys():
        value = path_tables[table]
        result = create_join_tree(value[0], path_tables)
        if type(result) is not list:
            result = [result]

        result.append(value[1])
        result.append(value[2])
        result.append(value[3])
        return result

    return table
