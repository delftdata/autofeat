import logging
import time

import pandas as pd
import tqdm

from feature_discovery.autofeat_pipeline.autofeat import AutoFeat
from feature_discovery.autofeat_pipeline.join_path_utils import get_path_length
from feature_discovery.config import RESULTS_FOLDER
from feature_discovery.experiments.result_object import Result
from feature_discovery.experiments.train_autogluon import run_auto_gluon
from feature_discovery.helpers.read_data import get_df_with_prefix

hyper_parameters = {"RF": {}, "GBM": {}, "XGB": {}, "XT": {}}


def evaluate_paths(bfs_result: AutoFeat, top_k: int, feat_sel_time: float, problem_type: str, top_k_paths: int = 15):
    logging.debug(f"Evaluate top-{top_k_paths} paths ... ")
    sorted_paths = sorted(bfs_result.ranking.items(), key=lambda r: (r[1], -get_path_length(r[0])), reverse=True)
    top_k_path_list = sorted_paths if len(sorted_paths) < top_k_paths else sorted_paths[:top_k_paths]
    base_features = bfs_result.partial_join_selected_features[bfs_result.base_table_id]

    all_results = []
    for path in tqdm.tqdm(top_k_path_list):
        print(path)
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
        print(dataframe.shape)
        features = list(set(features).intersection(set(dataframe.columns)))

        if len(features) < 2:
            features = bfs_result.partial_join_selected_features[bfs_result.base_table_id]
            features.append(bfs_result.target_column)

        start = time.time()
        best_model, results = run_auto_gluon(approach=Result.TFD,
                                             dataframe=dataframe[features],
                                             target_column=bfs_result.target_column,
                                             data_label=bfs_result.base_table_label,
                                             join_name=join_name,
                                             algorithms_to_run=hyper_parameters,
                                             value_ratio=bfs_result.value_ratio,
                                             problem_type=problem_type)
        end = time.time()
        for result in results:
            result.feature_selection_time = feat_sel_time
            result.train_time = end - start
            result.total_time += feat_sel_time
            result.total_time += end - start
            result.rank = rank
            result.top_k = top_k
            result.data_path = path_list

        all_results.extend(results)
        dataframe = None

    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / f"{bfs_result.base_table_label}_tfd.csv", index=False)
    return all_results, top_k_path_list


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
    # print(f"\t{table}")
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
