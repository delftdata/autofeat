import math
from typing import List

from feature_discovery.config import JOIN_RESULT_FOLDER
from feature_discovery.data_preparation.dataset_base import Dataset
from feature_discovery.data_preparation.join_data import prune_or_join_2
from feature_discovery.data_preparation.utils import get_paths, get_path_length
from feature_discovery.feature_selection.util_functions import (
    compute_correlation,
    compute_relevance_redundancy,
)
from feature_discovery.graph_processing.neo4j_transactions import (
    get_node_by_source_name,
    get_node_by_id,
)
from feature_discovery.helpers.util_functions import (
    get_elements_higher_than_value,
    transform_node_to_dict,
    objects_to_dict,
)


class Rank:
    def __init__(self, path, score, features):
        self.path = path
        self.score = score
        self.features = features


class Ranking:
    def __init__(self, dataset: Dataset, redundancy_threshold=20, cutoff_threshold=0.3):
        self.ranked_paths: List[Rank] = []
        self.joined_tables = {}
        self.dataset = dataset
        self.dataset.set_features()
        self.redundancy_threshold = redundancy_threshold
        self.cutoff_threshold = cutoff_threshold
        self.all_paths = get_paths()

    def start_ranking(self):
        path = []
        visited = set()
        self.ranking_recursion(self.dataset.base_table_id, path, visited)
        self.ranked_paths = sorted(
            self.ranked_paths,
            key=lambda r: (r.score, len(r.features), -get_path_length(r.path)),
            reverse=True,
        )
        print(objects_to_dict(self.ranked_paths))
        return self

    def ranking_recursion(self, base_table_id: str, path: list, visited: set):
        base_table_node = get_node_by_source_name(base_table_id)
        features = [
            transform_node_to_dict(node)
            for node in base_table_node
            if not node.get("name") == self.dataset.target_column
        ]

        for column in features:
            if column["id"] not in self.all_paths:
                continue

            print(f"Column: {column['label']}")
            visited.add(column["source_name"])
            for related_column in self.all_paths[column["id"]]:
                node = get_node_by_id(related_column)
                related_node = transform_node_to_dict(node)

                if related_node["source_name"] in visited:
                    continue

                print(f"\tRelated to: {related_node['label']}")
                ranked_path = self.ranking_func_2(column, related_node, path, visited)
                if ranked_path:
                    self.ranked_paths.append(ranked_path)
                    visited.remove(related_node["source_name"])
            if column["label"] in path:
                path.remove(column["label"])
        print(f"Processed all columns of table: {base_table_id}")
        return path

    def ranking_func_2(self, left_node, right_node, path: list, visited):
        if left_node["label"] not in path:
            path.append(left_node["label"])
        selected_features = []
        left_path = left_node["source_path"]
        left_node_name = left_node["source_name"]

        # If we already joined the tables on the path, we retrieve the join result
        if "--".join(path) in self.joined_tables:
            partial_join_path, selected_features = self.joined_tables["--".join(path)]
            left_path = partial_join_path
            left_node_name = partial_join_path.partition(f"../{JOIN_RESULT_FOLDER}/")[2]
            print(f"\t\tPartial join: {partial_join_path}")

        # Recursion logic
        # 1. Join existing left table with the current table visiting
        join_result_name = f"{left_node_name}--{right_node['source_name']}".replace(
            "/", "--"
        )
        result = prune_or_join_2(
            left_path,
            right_node["source_path"],
            left_node["name"],
            right_node["name"],
            join_result_name,
        )
        if result is None:
            return None

        joined_path, joined_df, left_table_df = result

        # Get the features from base table excluding target
        left_table_features = list(
            left_table_df.drop(columns=[self.dataset.target_column]).columns
        )
        join_table_features = list(
            joined_df.drop(columns=[self.dataset.target_column]).columns
        )
        features_right = [
            feat for feat in join_table_features if feat not in left_table_features
        ]

        # 2. Select dependent features
        dependent_features_scores = compute_correlation(
            left_table_features, joined_df, self.dataset.target_column
        )

        # 3. Compute the scores of the new features
        # Case 1: Foreign features vs selected features
        foreign_features_scores_1 = compute_relevance_redundancy(
            selected_features + self.dataset.base_table_features,
            features_right,
            joined_df,
            self.dataset.target_column,
            self.redundancy_threshold,
        )

        # 4. Select only the top uncorrelated features which are in the dependent features list
        score, features = _compute_ranking_score(
            dependent_features_scores, foreign_features_scores_1, self.cutoff_threshold
        )
        print(f"Path score: {score} and selected features\n\t{features}")
        # TODO: Future work
        # # Case 2: Foreign features vs base table features only
        # foreign_features_scores_2 = select_uncorrelated_with_selected(base_table_features, features_right,
        #                                                               joined_df, target_column)
        #
        # # 4. Select only the top uncorrelated features which are in the dependent features list
        # score_2, features_2 = _compute_ranking_score(dependent_features_scores, foreign_features_scores_2)

        # 5. Save the join for future reference/usage
        # features, score = (features_1, score_1) if score_1 < score_2 else (features_2, score_2)
        path_copy = path.copy()
        path_copy.append(right_node["label"])
        self.joined_tables["--".join(path_copy)] = (
            str(joined_path),
            selected_features + features,
        )

        print(f"Initiating recursion for {right_node['label']}")
        path = self.ranking_recursion(right_node["source_path"], path, visited)
        print(f"Exiting recursion with {path} \n\tscore: {score}")
        path.pop()
        rank = Rank("--".join(path_copy), score, features + selected_features)
        # rank = {"--".join(path_copy): (score, features + selected_features)}

        return rank


def _compute_ranking_score(
    dependent_features_scores: dict, foreign_features_scores: dict, cutoff_threshold
) -> tuple:
    # Create dataframe and normalise the data
    # normalised_dfs = normalize_dict_values(dependent_features_scores)
    # normalised_ffs = normalize_dict_values(foreign_features_scores)

    # Compute a score for each feature based on the dependent score and the foreign feature score
    # feat_scores = {feat: normalised_dfs[feat] + normalised_ffs[feat]
    feat_scores = {
        feat: 2 * dependent_features_scores[feat] + foreign_features_scores[feat]
        for feat in foreign_features_scores.keys()
        if feat in dependent_features_scores
    }

    # normalised_fs = normalize_dict_values(feat_scores)
    # Sort the features ascending based on the score
    # print(f"Normalised score on features:\n\t{normalised_fs}")
    selected_features = get_elements_higher_than_value(feat_scores, cutoff_threshold)
    selected_features = dict(
        sorted(selected_features.items(), key=lambda item: item[1], reverse=True)
    )
    # ns = normalize_dict_values(selected_features)
    # Assign a score to each join path for ranking
    # score = statistics.mean(selected_features.values()) if selected_features else -math.inf
    if selected_features:
        max_feat = max(selected_features, key=lambda item: item[0])
        score = selected_features[max_feat]
    else:
        score = -math.inf
    return score, list(selected_features.keys())
