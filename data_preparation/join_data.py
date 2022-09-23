import json
import os
from collections import Counter

import pandas as pd

from utils.file_naming_convention import JOIN_RESULT_FOLDER, MAPPING_FOLDER, ENUMERATED_PATHS
from utils.neo4j_utils import get_relation_properties, get_pk_fk_nodes
from utils.util_functions import transform_node_to_dict

folder_name = os.path.abspath(os.path.dirname(__file__))


def join_tables_recursive(all_paths: dict, mapping, current_table, target_column, path, allp, join_result_path,
                          joined_mapping=None):
    if not joined_mapping:
        joined_mapping = {}

    print(f"Current table: {current_table}")

    # Join and save the join result
    if not path == "":
        # get the name of the table from the path we created
        left_table = path.split("--")[-1]
        # get the location of the table we want to join with
        partial_join = mapping[left_table]

        # If we already joined the tables on the path, we retrieve the join result
        if path in joined_mapping:
            partial_join = joined_mapping[path]

        # Add the current table to the path
        path = f"{path}--{current_table}"
        joined_path, _, _ = join_and_save(partial_join, mapping[left_table],
                                          mapping[current_table], join_result_path, path)
        joined_mapping[path] = joined_path
    else:
        # Just started traversing, the path is the current table
        path = current_table

    print(path)
    allp.append(path)

    # Depth First Search recursively
    for table in all_paths[current_table]:
        # Break the cycles in the data, only visit new nodes
        if table not in path:
            join_path = join_tables_recursive(all_paths, mapping, table, target_column, path, allp, join_result_path,
                                              joined_mapping)
            # print(f"{join_path}")
    return path


def join_and_save(partial_join_path, left_table_path, right_table_path, join_result_path, join_result_name):
    # Getting the join keys
    from_col, to_col = get_relation_properties(left_table_path, right_table_path)
    # Read left side table
    left_table_df = pd.read_csv(os.path.join(folder_name, "../", partial_join_path), header=0, engine="python",
                                encoding="utf8", quotechar='"', escapechar='\\')
    if from_col not in left_table_df.columns:
        print(f"ERROR! Key {from_col} not in table {partial_join_path}")
        return None

    right_table_df = pd.read_csv(os.path.join(folder_name, "../", right_table_path), header=0, engine="python",
                                 encoding="utf8", quotechar='"', escapechar='\\')
    if to_col not in right_table_df.columns:
        print(f"ERROR! Key {to_col} not in table {right_table_path}")
        return None

    print(f"\tJoining {partial_join_path} with {right_table_path}\n\tOn keys: {from_col} - {to_col}")
    joined_df = pd.merge(left_table_df, right_table_df, how="left", left_on=from_col, right_on=to_col,
                         suffixes=("_b", ""))

    # If both tables have the same column, drop one of them
    duplicate_col = [col for col in joined_df.columns if col.endswith('_b')]
    # Drop the FK key from the left table
    # duplicate_col.append(to_col)
    joined_df.drop(columns=duplicate_col, inplace=True)
    # Save join result
    joined_path = f"{os.path.join(folder_name, '../', join_result_path)}/{join_result_name}"
    joined_df.to_csv(joined_path, index=False)

    return joined_path, joined_df, left_table_df


def join_all(base_table_id: str):
    queue = []
    queue.append(base_table_id)

    partial_join = None
    foreign_keys = []
    visited = set()

    mapping_columns = {}

    while len(queue) > 0:
        current_table = queue.pop()
        nodes = get_pk_fk_nodes(current_table)
        visited.add(current_table)
        primary_keys = set()
        for pk, fk in nodes:
            pk_node = transform_node_to_dict(pk)
            fk_node = transform_node_to_dict(fk)

            left_on = pk_node['name']
            if pk_node['source_path'] in mapping_columns:
                left_source = mapping_columns[pk_node['source_path']]
                if pk_node['name'] in left_source:
                    left_on = left_source[pk_node['name']]

            if not left_on == fk_node['name']:
                primary_keys.add(left_on)

            if fk_node['source_path'] in visited:
                continue
            else:
                visited.add(fk_node['source_path'])

            queue.append(fk_node['source_path'])

            left_table = pd.read_csv(pk_node['source_path'])
            right_table = pd.read_csv(fk_node['source_path'])
            if partial_join is not None:
                left_table = partial_join

            partial_join = pd.merge(left_table, right_table, how="left", left_on=left_on,
                                    right_on=fk_node['name'], suffixes=("", fk_node['source_name']))
            modified_columns = {col.partition(fk_node['source_name'])[0]: col for col in partial_join.columns if
                                col.endswith(fk_node['source_name'])}
            if fk_node['name'] in modified_columns:
                foreign_keys.append(modified_columns[fk_node['name']])
            else:
                foreign_keys.append(fk_node['name'])

            mapping = {fk_node['source_path']: modified_columns}
            mapping_columns.update(mapping)
            # duplicate_col = [col for col in partial_join.columns if col.endswith('_b')]
            # partial_join.drop(columns=duplicate_col, inplace=True)
        partial_join.drop(columns=list(primary_keys), inplace=True)

    partial_join.drop(columns=list(set(foreign_keys).intersection(set(partial_join.columns))), inplace=True)

    return partial_join


def prune_or_join_2(left_path, right_path, left_key, right_key, join_result_name, prune_threshold=0.3):
    # Read left side table
    left_table_df = pd.read_csv(left_path, header=0, engine="python", encoding="utf8", quotechar='"', escapechar='\\')
    right_table_df = pd.read_csv(right_path, header=0, engine="python", encoding="utf8", quotechar='"', escapechar='\\')

    # Verify join quality
    print(f"Verifying join quality:\n\t{left_path}\n\t\t{left_key}\n\t{right_path}\n\t\t{right_key}")
    result = prune_table(left_table_df[left_key], right_table_df[right_key], prune_threshold)
    if result:
        print("Null value ratio exceeding threshold. Pruning the table ... ")
        return None

    # Test join scenario 1:N - aggregate, M:N - prune
    right_table = join_scenario(left_table_df, right_table_df, left_key, right_key)
    if right_table is None:
        return None

    print(f"\tJoining {left_path} with {right_path}\n\tOn keys: {left_key} - {right_key}")
    joined_df = pd.merge(left_table_df, right_table, how="left", left_on=left_key, right_on=right_key,
                         suffixes=("_b", ""))

    # If both tables have the same column, drop one of them
    duplicate_col = [col for col in joined_df.columns if col.endswith('_b')]
    joined_df.drop(columns=duplicate_col, inplace=True)
    # Save join result
    joined_path = f"../{JOIN_RESULT_FOLDER}/{join_result_name}"
    joined_df.to_csv(joined_path, index=False)

    return joined_path, joined_df, left_table_df


def prune_or_join(partial_join_path, left_table_name, right_table_name, mapping, join_result_path, prune_threshold=0.3):
    # Getting the join keys
    left_tokens = left_table_name.split('/')
    right_tokens = right_table_name.split('/')

    left_name = '-'.join(left_tokens[0:-1]).partition('.csv')[0]
    right_name = '-'.join(right_tokens[0:-1])

    left_key = left_tokens[-1]
    right_key = right_tokens[-1]

    # Read left side table
    left_table_df = pd.read_csv(os.path.join(folder_name, partial_join_path), header=0, engine="python",
                                encoding="utf8", quotechar='"', escapechar='\\')
    if left_key not in left_table_df.columns:
        print(f"ERROR! Key {left_key} not in table {partial_join_path}")
        return None

    right_table_df = pd.read_csv(os.path.join(folder_name, mapping[right_table_name]), header=0, engine="python",
                                 encoding="utf8", quotechar='"', escapechar='\\')
    if right_key not in right_table_df.columns:
        print(f"ERROR! Key {right_key} not in table {right_table_name}")
        return None

    # Verify join quality
    print(f"Verifying join quality:\n\t{partial_join_path}\n\t\t{left_key}\n\t{right_table_name}\n\t\t{right_key}")
    result = prune_table(left_table_df[left_key], right_table_df[right_key], prune_threshold)
    if result:
        print("Null value ratio exceeding threshold. Pruning the table ... ")
        return None

    # Test join scenario 1:N - aggregate, M:N - prune
    right_table = join_scenario(left_table_df, right_table_df, left_key, right_key)
    if right_table is None:
        return None

    print(f"\tJoining {partial_join_path} with {right_table_name}\n\tOn keys: {left_key} - {right_key}")
    joined_df = pd.merge(left_table_df, right_table, how="left", left_on=left_key, right_on=right_key,
                         suffixes=("_b", ""))

    # If both tables have the same column, drop one of them
    duplicate_col = [col for col in joined_df.columns if col.endswith('_b')]
    joined_df.drop(columns=duplicate_col, inplace=True)
    # Save join result
    joined_path = f"{os.path.join(folder_name, '../', join_result_path)}/{left_name}--{right_name}"
    joined_df.to_csv(joined_path, index=False)

    return joined_path, joined_df, left_table_df


def prune_table(left_column, right_column, null_threshold=0.3):
    set_intersection = set(left_column).intersection(set(right_column))
    if len(set_intersection) == 0:
        return True

    set_difference = list(set(left_column) - set(right_column))
    null_ratio = sum([Counter(left_column)[el] for el in set_difference]) / len(left_column)

    print(f"Null values ratio: {null_ratio}")
    if null_ratio > null_threshold:
        return True

    return False


def join_scenario(left_df, right_df, left_join_col, right_join_col):
    left_count = Counter(left_df[left_join_col])
    right_count = Counter(right_df[right_join_col])

    if any([el > 1 for el in left_count.values()]) and any([el > 1 for el in right_count.values()]):
        print("M:N scenario not supported, pruning table ... ")
        return None
    elif any([el > 1 for el in right_count.values()]):
        print("1:N scenario -- aggregating right table ... ")
        right_join_dataframe = aggregate_rows(right_df, right_join_col)
        print(f"{len(right_df)} aggregated to {len(right_join_dataframe)}")
        return right_join_dataframe

    return right_df


def aggregate_rows(dataframe, join_column):
    indexes = []
    df = dataframe.apply(lambda x: pd.factorize(x)[0] if x.dtype == object else x)
    for value in df[join_column].unique():
        section = df.loc[df[join_column] == value]
        idmax_per_col = section.idxmax()
        chosen_idx, occ = Counter(idmax_per_col).most_common(1)[0]
        indexes.append(chosen_idx)

    return dataframe.loc[indexes, :]


def enumerate_all(base_table_id: str, all_paths: dict, path: list, enumerated_paths: list, visited: list):
    all_paths_keys = all_paths.keys()
    nodes = [k for k in all_paths_keys if base_table_id in k]

    visited.append(base_table_id)
    for node in nodes:
        if node not in path:
            path.append(node)
            enumerated_paths.append(path)

        related = all_paths[node]
        for rel in related:
            table_id = '/'.join(rel.split('/')[:-1])
            if table_id in visited:
                continue

            cp = path.copy()
            cp.append(rel)
            enumerated_paths.append(cp)

            path = enumerate_all(table_id, all_paths, cp.copy(), enumerated_paths, visited)

        path.pop()

    visited.pop()
    return path

