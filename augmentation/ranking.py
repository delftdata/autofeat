from augmentation.data_preparation_pipeline import join_and_save
from augmentation.pipeline import apply_feat_sel, classify_and_rank


def ranking_func(all_paths: dict, mapping, current_table, target_column, path, allp, join_result_folder_path,
                            ranking, joined_mapping: dict):

    # Join and save the join result
    if not path == "":
        # get the name of the table from the path we created
        left_table = path.split("--")[-1]
        # get the location of the table we want to join with
        partial_join_path = mapping[left_table]

        # If we already joined the tables on the path, we retrieve the join result
        if path in joined_mapping:
            partial_join_path = joined_mapping[path]

        # Add the current table to the path
        path = f"{path}--{current_table}"

        # Recursion logic
        # 1. Join existing left table with the current table visiting
        joined_path, joined_df, left_table_df = join_and_save(partial_join_path, mapping[left_table],
                                                              mapping[current_table], join_result_folder_path, path)
        # 2. Apply filter-based feature selection and normalise data
        new_features_ranks = apply_feat_sel(joined_df, left_table_df, target_column, path)
        # 3. Use the scores from the feature selection to predict the rank and add it to the ranking set
        result = classify_and_rank(new_features_ranks)
        ranking.update(result)
        # 4. Save the join for future reference/usage
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
            join_path = ranking_func(all_paths, mapping, table, target_column, path, allp, join_result_folder_path,
                                                ranking, joined_mapping)
    return path