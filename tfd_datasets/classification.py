from config import ROOT_FOLDER, DATA
from data_preparation.dataset_base import Dataset, DatasetType


# classification
school_small = Dataset(
    base_table_id=ROOT_FOLDER / DATA / "ARDA/school/base.csv",
    base_table_name="base.csv",
    base_table_label="ARDA",
    target_column="class",
    dataset_type=DatasetType.CLASSIFICATION,
    base_table_features=["DBN", "School Name", "School Type", "Total Parent Response Rate (%)",
                         "Total Teacher Response Rate (%)", "Total Student Response Rate (%)"]
)

accounting = Dataset(
    base_table_id=ROOT_FOLDER / DATA / "cs/target_churn.csv",
    base_table_name="target_churn.csv",
    base_table_label="cs",
    target_column="target_churn",
    dataset_type=DatasetType.CLASSIFICATION,
    base_table_features=["ACC_KEY", "date_horizon"]
)

nyc = Dataset(
    base_table_id=ROOT_FOLDER / DATA / "nydata/housing.csv",
    base_table_name="housing.csv",
    base_table_label="nydata",
    target_column="Reporting Construction Type",
    dataset_type=DatasetType.CLASSIFICATION,
    base_table_features=["Project ID", "Project Name", "Project Start Date", "Address", "Borough", "Postcode",
                         "Community Board", "NTA - Neighborhood Tabulation Area", "Prevailing Wage Status"]
)

# Not ingested yet
la_data = Dataset(
    base_table_id=ROOT_FOLDER / DATA / "nydata/housing.csv",
    base_table_name="housing.csv",
    base_table_label="ladata",
    target_column="Reporting Construction Type",
    dataset_type=DatasetType.CLASSIFICATION,
    base_table_features=["Project ID", "Project Name", "Project Start Date", "Address", "Borough", "Postcode",
                         "Community Board", "NTA - Neighborhood Tabulation Area", "Reporting Construction Type",
                         "Prevailing Wage Status"]
)

# Regression datasets
air = Dataset(
    base_table_id=ROOT_FOLDER / DATA / "air/temp.csv",
    base_table_name="temp.csv",
    base_table_label="air",
    target_column="Temperature",
    dataset_type=DatasetType.REGRESSION,
    base_table_features=["Day", "f0_", "f1_", "f2_", "f3_", "f4_", "f5_", "f6_"]
)
