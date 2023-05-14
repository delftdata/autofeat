

# Regression datasets
from feature_discovery.config import ROOT_FOLDER, DATA
from feature_discovery.data_preparation.dataset_base import Dataset, REGRESSION

air = Dataset(
    base_table_path=ROOT_FOLDER / DATA / "air",
    base_table_name="temp.csv",
    base_table_label="air",
    target_column="Temperature",
    dataset_type=REGRESSION,
    base_table_features=["Day", "f0_", "f1_", "f2_", "f3_", "f4_", "f5_", "f6_"]
)