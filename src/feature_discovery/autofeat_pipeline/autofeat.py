import logging
import tempfile
import uuid
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional

import pandas as pd
from autogluon.features.generators import AutoMLPipelineFeatureGenerator


from feature_discovery.autofeat_pipeline.join_data import join_and_save
from feature_discovery.autofeat_pipeline.join_path_feature_selection import RelevanceRedundancy
from feature_discovery.autofeat_pipeline.join_path_utils import compute_join_name
from feature_discovery.experiments.dataset_object import CLASSIFICATION
from feature_discovery.graph_processing.neo4j_transactions import (
    get_adjacent_nodes,
    get_relation_properties_node_name,
)
from feature_discovery.helpers.read_data import get_df_with_prefix

logging.getLogger().setLevel(logging.WARNING)


class AutoFeat:
    def __init__(
            self,
            base_table_label: str,
            base_table_id: str,
            target_column: str,
            task: str = CLASSIFICATION,
            value_ratio: float = 0.65,
            top_k: int = 5,
            sample_size: int = 3000,
            pearson: bool = False,
            jmi: bool = False,
            no_relevance: bool = False,
            no_redundancy: bool = False
    ):
        """

        :param base_table_label: The name (label) of the base table to be used for saving data.
        :param target_column: Target column containing the class labels for training.
        :param value_ratio: Pruning threshold. It represents the ration between the number of non-null values in a column and the total number of values.
        """
        self.base_table_label: str = base_table_label
        self.target_column: str = target_column
        self.value_ratio: float = value_ratio
        self.top_k: int = top_k
        self.sample_size: int = sample_size
        self.base_table_id: str = base_table_id
        self.task: str = task
        # Mapping with the name of the join and the corresponding name of the file containing the join result.
        self.join_name_mapping: Dict[str, str] = {}
        # Set used to track the visited nodes.
        self.discovered: Set[str] = set()
        # Save the selected features of the previous join path (used for conditional redundancy)
        self.partial_join_selected_features: Dict[str, List] = {}

        self.ranking: Dict[str, float] = {}
        self.join_keys: Dict[str, list] = {}
        self.rel_red = RelevanceRedundancy(target_column)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.partial_join = self.initialisation()

        # Ablation study parameters
        self.sample_data_step = True
        self.pearson = pearson
        self.jmi = jmi
        self.no_relevance = no_relevance
        self.no_redundancy = no_redundancy

    def initialisation(self):
        from sklearn.model_selection import train_test_split

        # Read dataframe
        base_table_df, partial_join_name = get_df_with_prefix(self.base_table_id, self.target_column)

        # Stratified sampling
        if self.sample_size < base_table_df.shape[0]:
            if self.task == CLASSIFICATION:
                X_train, X_test = train_test_split(base_table_df, train_size=self.sample_size,
                                                   stratify=base_table_df[self.target_column])
            else:
                X_train, X_test = train_test_split(base_table_df, train_size=self.sample_size)
        else:
            X_train = base_table_df

        # Base table features are the selected features
        features = list(X_train.columns)
        if self.target_column in features:
            features.remove(self.target_column)

        self.partial_join_selected_features[partial_join_name] = features
        self.ranking[partial_join_name] = 0
        self.join_keys[partial_join_name] = []

        return X_train

    def streaming_feature_selection(self, queue: set, previous_queue: set = None):
        if len(queue) == 0:
            return

        if previous_queue is None:
            previous_queue = queue.copy()

        # Iterate through all the elements of the queue:
        # 1) in the first iteration: queue = base_node_id
        # 2) in all the other iterations: queue = neighbours of the previous node
        all_neighbours = set()
        while len(queue) > 0:
            # Get the current/base node
            base_node_id = queue.pop()
            self.discovered.add(base_node_id)
            logging.debug(f"New iteration with base node: {base_node_id}")

            # Determine the neighbours (unvisited)
            neighbours = sorted(set(get_adjacent_nodes(base_node_id)) - set(self.discovered))
            if len(neighbours) == 0:
                continue

            all_neighbours.update(neighbours)

            # Process every neighbour - join, determine quality, get features
            for node in neighbours:
                self.discovered.add(node)
                logging.debug(f"Adjacent node: {node}")

                # Get the join keys with the highest score
                join_keys = get_relation_properties_node_name(from_id=base_node_id, to_id=node)
                if len(join_keys) == 1:
                    highest_ranked_join_keys = join_keys
                else:
                    highest_ranked_join_keys = []
                    for jk in join_keys:
                        if jk[0]['weight'] == join_keys[0][0]['weight']:
                            highest_ranked_join_keys.append(jk)
                        else:
                            break

                # Read the neighbour node
                right_df, right_label = get_df_with_prefix(node)
                logging.debug(f"\tRight table shape: {right_df.shape}")

                current_queue = set()
                while len(previous_queue) > 0:
                    previous_join_name = previous_queue.pop()

                    if previous_join_name == self.base_table_id:
                        previous_join_name = self.base_table_id
                        previous_join = self.partial_join.copy()
                    else:
                        previous_join = pd.read_csv(
                            Path(self.temp_dir.name) / self.join_name_mapping[previous_join_name],
                            header=0,
                            engine="python",
                            encoding="utf8",
                            quotechar='"',
                            escapechar='\\',
                        )

                    # The current node can only be joined through the base node.
                    # If the base node doesn't exist in the previous join path, the join can't be performed
                    if base_node_id not in previous_join_name:
                        logging.debug(f"\tBase node {base_node_id} not in partial join {previous_join_name}")
                        continue

                    for prop in highest_ranked_join_keys:
                        join_prop, from_table, to_table = prop
                        if join_prop['from_label'] != from_table:
                            continue

                        if join_prop['from_column'] == self.target_column:
                            current_queue.add(previous_join_name)
                            continue

                        logging.debug(f"\t\tJoin properties: {join_prop}")

                        # Step - Explore all possible join paths based on the join keys - Compute the name of the join
                        join_name = compute_join_name(join_key_property=prop, partial_join_name=previous_join_name)
                        logging.debug(f"\tJoin name: {join_name}")

                        # Step - Join
                        joined_df, join_filename, join_columns = self.step_join(join_key_properties=prop,
                                                                                left_df=previous_join,
                                                                                right_df=right_df,
                                                                                right_label=right_label)
                        if joined_df is None:
                            current_queue.add(previous_join_name)
                            continue

                        data_quality = self.step_data_quality(join_key_properties=prop, joined_df=joined_df)
                        if not data_quality:
                            current_queue.add(previous_join_name)
                            continue

                        result = self.streaming_relevance_redundancy(dataframe=joined_df,
                                                                     new_features=list(right_df.columns),
                                                                     selected_features=
                                                                     self.partial_join_selected_features[
                                                                         previous_join_name],
                                                                     no_relevance=self.no_relevance,
                                                                     no_redundancy=self.no_redundancy,
                                                                     )
                        if result is not None:
                            self.ranking[join_name] = result[0]
                            all_selected_features = self.partial_join_selected_features[
                                previous_join_name]
                            all_selected_features.extend(result[1])
                            self.partial_join_selected_features[join_name] = all_selected_features
                        else:
                            self.partial_join_selected_features[join_name] = self.partial_join_selected_features[
                                previous_join_name]

                        join_columns.extend(self.join_keys[previous_join_name])
                        self.join_keys[join_name] = join_columns
                        self.join_name_mapping[join_name] = join_filename

                        current_queue.add(join_name)
                # Initialise the queue with the new paths (current_queue)
                previous_queue.update(current_queue)
        self.streaming_feature_selection(all_neighbours, previous_queue)

    def streaming_relevance_redundancy(self, dataframe: pd.DataFrame,
                                       new_features: List[str],
                                       selected_features: List[str],
                                       no_relevance: bool = False,
                                       no_redundancy: bool = False) -> Optional[Tuple[float, List[dict]]]:

        # df = AutoMLPipelineFeatureGenerator(
        #     enable_text_special_features=False, enable_text_ngram_features=False
        # ).fit_transform(X=dataframe)

        X = dataframe.drop(columns=[self.target_column])
        y = dataframe[self.target_column]

        features = list(set(X.columns).intersection(set(new_features)))
        top_feat = len(features) if len(features) < self.top_k else self.top_k

        relevant_features = new_features
        sum_m = 0
        m = 1
        if not no_relevance:
            feature_score_relevance = self.rel_red.measure_relevance(dataframe=X,
                                                                     new_features=features,
                                                                     target_column=y,
                                                                     pearson=self.pearson)[:top_feat]
            if len(feature_score_relevance) == 0:
                return None
            relevant_features = list(dict(feature_score_relevance).keys())
            m = len(feature_score_relevance) if len(feature_score_relevance) > 0 else m
            sum_m = sum(list(map(lambda x: x[1], feature_score_relevance)))

        final_features = relevant_features
        sum_o = 0
        o = 1
        if not no_redundancy:
            feature_score_redundancy = self.rel_red.measure_redundancy(dataframe=X,
                                                                       selected_features=selected_features,
                                                                       relevant_features=relevant_features,
                                                                       target_column=y,
                                                                       jmi=self.jmi)

            if len(feature_score_redundancy) == 0:
                return None

            o = len(feature_score_redundancy) if feature_score_redundancy else o
            sum_o = sum(list(map(lambda x: x[1], feature_score_redundancy)))
            final_features = list(dict(feature_score_redundancy).keys())

        score = (o * sum_m + m * sum_o) / (m * o)

        return score, final_features

    def step_join(
            self, join_key_properties: tuple, left_df: pd.DataFrame, right_df: pd.DataFrame, right_label: str
    ) -> Tuple[pd.DataFrame or None, str, list]:
        logging.debug("\tSTEP Join ... ")
        join_prop, from_table, to_table = join_key_properties

        # Step - Sample neighbour data - Transform to 1:1 or M:1
        sampled_right_df = right_df
        if self.sample_data_step:
            sampled_right_df = right_df.groupby(f"{right_label}.{join_prop['to_column']}").sample(n=1, random_state=42)

        # File naming convention as the filename can be gigantic
        join_filename = f"{self.base_table_label}_join_BFS_{self.value_ratio}_{str(uuid.uuid4())}.csv"

        # Join
        left_on = f"{from_table}.{join_prop['from_column']}"
        right_on = f"{to_table}.{join_prop['to_column']}"
        joined_df = join_and_save(
            left_df=left_df,
            right_df=sampled_right_df,
            left_column_name=left_on,
            right_column_name=right_on,
            join_path=Path(self.temp_dir.name) / join_filename,
        )
        if joined_df is None:
            return None, join_filename, []

        return joined_df, join_filename, [left_on, right_on]

    def step_data_quality(self, join_key_properties: tuple, joined_df: pd.DataFrame) -> bool:
        logging.debug("\tSTEP data quality ...")
        join_prop, _, to_table = join_key_properties

        # Data Quality check - Prune the joins with high null values ratio
        if joined_df[f"{to_table}.{join_prop['to_column']}"].count() / joined_df.shape[0] < self.value_ratio:
            logging.debug(f"\t\tRight column value ration below {self.value_ratio}.\nSKIPPED Join")
            return False

        return True
