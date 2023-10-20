import logging
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Set, Tuple, Optional

import pandas as pd

from feature_discovery.autofeat_pipeline.join_data import join_and_save
from feature_discovery.autofeat_pipeline.join_path_utils import compute_join_name
from feature_discovery.graph_processing.neo4j_transactions import get_adjacent_nodes, get_relation_properties_node_name
from feature_discovery.helpers.read_data import get_df_with_prefix


class JoinAll:
    def __init__(
            self,
            base_table_id: str,
            target_column: str,
    ):
        """

        :param base_table_label: The name (label) of the base table to be used for saving data.
        :param target_column: Target column containing the class labels for training.
        :param value_ratio: Pruning threshold. It represents the ration between the number of non-null values in a column and the total number of values.
        """
        self.target_column: str = target_column
        self.base_table_id: str = base_table_id
        # Mapping with the name of the join and the corresponding name of the file containing the join result.
        self.join_name_mapping: Dict[str, str] = {}
        # Set used to track the visited nodes.
        self.discovered: Set[str] = set()
        self.join_keys: Dict[str, list] = {}
        self.temp_dir = tempfile.TemporaryDirectory()
        self.partial_join_name: Optional[str] = None

        self.partial_join = self.initialisation()

    def initialisation(self):
        # Read dataframe
        base_table_df, partial_join_name = get_df_with_prefix(self.base_table_id, self.target_column)
        self.join_keys[partial_join_name] = []
        self.partial_join_name = partial_join_name

        return base_table_df

    def join_all_bfs(self, queue: set):
        if len(queue) == 0:
            previous_join = pd.read_csv(
                Path(self.temp_dir.name) / self.join_name_mapping[self.partial_join_name],
                header=0,
                engine="python",
                encoding="utf8",
                quotechar='"',
                escapechar='\\',
            )
            return previous_join

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

                # Read the neighbour node
                right_df, right_label = get_df_with_prefix(node)
                logging.debug(f"\tRight table shape: {right_df.shape}")

                # Get the data which has been joined so far
                if self.partial_join_name == self.base_table_id:
                    previous_join = self.partial_join.copy()
                else:
                    previous_join = pd.read_csv(
                        Path(self.temp_dir.name) / self.join_name_mapping[self.partial_join_name],
                        header=0,
                        engine="python",
                        encoding="utf8",
                        quotechar='"',
                        escapechar='\\',
                    )

                # The current node can only be joined through the base node.
                # If the base node doesn't exist in the previous join path, the join can't be performed
                if base_node_id not in self.partial_join_name:
                    logging.debug(f"\tBase node {base_node_id} not in partial join {self.partial_join_name}")
                    continue

                join_name = None
                for prop in join_keys:
                    join_prop, from_table, to_table = prop
                    if join_prop['from_label'] != from_table:
                        continue

                    if join_prop['from_column'] == self.target_column:
                        continue

                    logging.debug(f"\t\tJoin properties: {join_prop}")

                    # Step - Explore all possible join paths based on the join keys - Compute the name of the join
                    join_name = compute_join_name(join_key_property=prop, partial_join_name=self.partial_join_name)
                    logging.debug(f"\tJoin name: {join_name}")

                    # Step - Join
                    joined_df, join_filename, join_columns = self.step_join(join_key_properties=prop,
                                                                            left_df=previous_join,
                                                                            right_df=right_df,
                                                                            right_label=right_label)
                    if joined_df is None:
                        continue

                    join_columns.extend(self.join_keys[self.partial_join_name])
                    self.join_keys[join_name] = list(set(join_columns))
                    self.join_name_mapping[join_name] = join_filename
                    self.partial_join_name = join_name
                    break

                # If the
                if join_name and self.partial_join_name != join_name:
                    all_neighbours.remove(node)

        return self.join_all_bfs(all_neighbours)

    def step_join(
            self, join_key_properties: tuple, left_df: pd.DataFrame, right_df: pd.DataFrame, right_label: str
    ) -> Tuple[pd.DataFrame or None, str, list]:
        logging.debug("\tSTEP Join ... ")
        join_prop, from_table, to_table = join_key_properties

        # Step - Sample neighbour data - Transform to 1:1 or M:1
        sampled_right_df = right_df.groupby(f"{right_label}.{join_prop['to_column']}").sample(n=1, random_state=42)

        # File naming convention as the filename can be gigantic
        join_filename = f"join_BFS_{str(uuid.uuid4())}.csv"

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
