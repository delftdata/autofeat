from typing import List, Optional

import pandas as pd


CLASSIFICATION = "false"
REGRESSION = "true"


class Dataset:
    def __init__(self, base_table_id: str, base_table_name: str, base_table_label: str, target_column: str,
                 dataset_type: bool, base_table_features: Optional[List] = None):
        self.base_table_id = base_table_id
        self.target_column = target_column
        self.base_table_name = base_table_name
        self.base_table_label = base_table_label
        self.base_table_features = base_table_features
        self.dataset_type = dataset_type
        self.base_table_df = None

    def set_features(self):
        if self.base_table_df is not None:
            self.base_table_features = list(self.base_table_df.drop(columns=[self.target_column]).columns)
        else:
            self.base_table_features = list(
                pd.read_csv(self.base_table_id, header=0, engine="python", encoding="utf8", quotechar='"',
                            escapechar='\\', nrows=1).drop(columns=[self.target_column]).columns)

    def set_base_table_df(self):
        self.base_table_df = pd.read_csv(self.base_table_id, header=0, engine="python", encoding="utf8", quotechar='"',
                                         escapechar='\\')

