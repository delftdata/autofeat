from config import ROOT_FOLDER, DATA
from data_preparation.dataset_base import Dataset, CLASSIFICATION

# classification
school_small = Dataset(
    base_table_path=ROOT_FOLDER / DATA / "school",
    base_table_name="base.csv",
    base_table_label="school",
    target_column="class",
    dataset_type=CLASSIFICATION,
    base_table_features=["DBN", "School Name", "School Type", "Total Parent Response Rate (%)",
                         "Total Teacher Response Rate (%)", "Total Student Response Rate (%)"]
)

accounting = Dataset(
    base_table_path=ROOT_FOLDER / DATA / "accounting",
    base_table_name="target_churn.csv",
    base_table_label="accounting",
    target_column="target_churn",
    dataset_type=CLASSIFICATION,
    base_table_features=["ACC_KEY", "date_horizon"]
)

nyc = Dataset(
    base_table_path=ROOT_FOLDER / DATA / "nydata",
    base_table_name="housing.csv",
    base_table_label="nyc",
    target_column="Reporting Construction Type",
    dataset_type=CLASSIFICATION,
    base_table_features=["Project ID", "Project Name", "Project Start Date", "Address", "Borough", "Postcode",
                         "Community Board", "NTA - Neighborhood Tabulation Area", "Prevailing Wage Status"]
)

credit = Dataset(
    base_table_path=ROOT_FOLDER / DATA / "credit",
    base_table_name="table8.csv",
    base_table_label="credit",
    target_column="class",
    dataset_type=CLASSIFICATION,
    base_table_features=["age", "credit_amount"]
)

steel = Dataset(
    base_table_path=ROOT_FOLDER / DATA / "steel",
    base_table_name="table10.csv",
    base_table_label="steel",
    target_column="Class",
    dataset_type=CLASSIFICATION,
    base_table_features=["V13", "V14", "V27", "V28", "V29", "V3", "V32"]
)

cylinder = Dataset(
    base_table_path=ROOT_FOLDER / DATA / "cylinder",
    base_table_name="table2.csv",
    base_table_label="cylinder",
    target_column="band_type",
    dataset_type=CLASSIFICATION,
    base_table_features=["anode_space_ratio", "blade_pressure", "caliper", "customer", "cylinder_number",
                         "grain_screened", "hardener", "humifity", "ink_temperature", "ink_type", "plating_tank",
                         "press_speed", "proof_cut", "roughness", "timestamp", "type_on_cylinder", "unit_number",
                         "varnish_pct", "wax"]
)

# Not ingested yet
la_data = Dataset(
    base_table_path=ROOT_FOLDER / DATA / "ladata",
    base_table_name="housing.csv",
    base_table_label="ladata",
    target_column="Reporting Construction Type",
    dataset_type=CLASSIFICATION,
    base_table_features=["Project ID", "Project Name", "Project Start Date", "Address", "Borough", "Postcode",
                         "Community Board", "NTA - Neighborhood Tabulation Area", "Reporting Construction Type",
                         "Prevailing Wage Status"]
)
