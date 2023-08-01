from pathlib import Path

import pandas as pd

from feature_discovery.config import DATA_FOLDER
from feature_discovery.experiments.dataset_object import Dataset

CLASSIFICATION_DATASETS = []
REGRESSION_DATASETS = []
ALL_DATASETS = []


def init_datasets():
    print("Initialising datasets ...")
    datasets_df = pd.read_csv(DATA_FOLDER / "datasets.csv")

    for index, row in datasets_df.iterrows():
        dataset = Dataset(base_table_label=row["base_table_label"],
                          target_column=row["target_column"],
                          base_table_path=Path(row["base_table_path"]),
                          base_table_name=row["base_table_name"],
                          dataset_type=row["dataset_type"])
        if row["dataset_type"] == "regression":
            REGRESSION_DATASETS.append(dataset)
        else:
            CLASSIFICATION_DATASETS.append(dataset)

    ALL_DATASETS.extend(CLASSIFICATION_DATASETS)
    ALL_DATASETS.extend(REGRESSION_DATASETS)
