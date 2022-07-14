import os
from collections import Counter

import pandas as pd

from utils.neo4j_utils import get_relation_properties

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

    print(f"\tJoining {partial_join_path} with {right_table_name}\n\tOn keys: {left_key} - {right_key}")

    # Verify join quality
    result = prune_table(left_table_df[left_key], right_table_df[right_key], prune_threshold)
    if result:
        return None

    # Test join scenario 1:N - aggregate, M:N - prune
    right_table = join_scenario(left_table_df, right_table_df, left_key, right_key)
    if right_table is None:
        return None

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
    count_values = Counter(left_column)

    null_rows = sum([count_values[el] for el in set_difference])/len(left_column)

    if null_rows > null_threshold:
        return True

    return False


def join_scenario(left_df, right_df, left_join_col, right_join_col):
    left_count = Counter(left_df[left_join_col])
    right_count = Counter(right_df[right_join_col])

    if any([el > 1 for el in left_count.values()]) and any([el > 1 for el in right_count.values()]):
        return None
    elif any([el > 1 for el in right_count.values()]):
        right_join_dataframe = aggregate_rows(right_df, right_join_col)
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







