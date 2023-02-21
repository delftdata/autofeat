from typing import Dict

from config import DATA_FOLDER
from data_preparation.dataset_base import Dataset


class Datasets:
    titanic = Dataset(
        base_table_id=DATA_FOLDER / "titanic/titanic.csv",
        base_table_name="titanic.csv",
        base_table_label="titanic",
        target_column="Survived"
    )

    steel_plate_fault = Dataset(
        base_table_id=DATA_FOLDER / "steel-plate-fault/steel_plate_fault.csv",
        base_table_name="steel_plate_fault.csv",
        base_table_label="steel_plate_fault",
        target_column="Class"
    )

    football = Dataset(
        base_table_id=DATA_FOLDER / "football/football.csv",
        base_table_name="football.csv",
        base_table_label="football",
        target_column="win"
    )

    school = Dataset(
        base_table_id=DATA_FOLDER / "school" / "base.csv",
        base_table_name="base.csv",
        base_table_label="school_small",
        target_column="class"
    )

    ALL = [titanic, steel_plate_fault, football]

    titanic_data: Dict[str, str] = {
        "join_result_folder_path": "joined-df/titanic",
        "label_column": "Survived",
        "base_table_name": "titanic.csv",
        "base_table_label": "titanic",
        "path": "other-data/decision-trees-split/titanic",
        "mappings_folder_name": "mappings/titanic",
        "id": "../other-data/synthetic/titanic/titanic.csv"
    }

    steel_data: Dict[str, str] = {
        "join_result_folder_path": "joined-df/steel-plate-fault",
        "label_column": "Class",
        "base_table_name": "steel_plate_fault.csv",
        "base_table_label": "steel_plate_fault",
        "path": "other-data/decision-trees-split/steel-plate-fault",
        "mappings_folder_name": "mappings/steel-plate-fault",
        "id": "../other-data/synthetic/steel-plate-fault/steel_plate_fault.csv"
    }

    football_data: Dict[str, str] = {
        "join_result_folder_path": "joined-df/football",
        "label_column": "win",
        "base_table_name": "football.csv",
        "base_table_label": "football",
        "path": "other-data/decision-trees-split/football",
        "mappings_folder_name": "mappings/football",
        "id": "../other-data/synthetic/football/football.csv"
    }

    # citeSeer_data: Dict[str, str] = {
    #     "path": "other-data/data/CiteSeer",
    #     "mappings_folder_name": "mappings/pub/CiteSeer",
    #     "join_result_folder_path": "joined-df/pub/CiteSeer",
    #     "label_column": "class_label",
    #     "base_table_name": "paper.csv",
    # }
    #
    # cora_data: Dict[str, str] = {
    #     "path": "other-data/data/CORA",
    #     "mappings_folder_name": "mappings/pub/CORA",
    #     "join_result_folder_path": "joined-df/pub/CORA",
    #     "label_column": "class_label",
    #     "base_table_name": "paper.csv",
    # }

    # Used with the data discovery scenario, where multiple datasets are together
    # PubMed contains: CORA, PubMed_Diabetes, WebKP, CiteSeer

    # pubmed_data: Dict[str, str] = {
    #     "path": "other-data/data/PubMed_Diabetes",
    #     "mappings_folder_name": "mappings/pub/PubMed_Diabetes",
    #     "join_result_folder_path": "joined-df/pub/PubMed_Diabetes",
    #     "label_column": "class_label",
    #     "base_table_name": "paper.csv",
    # }

    # web_data: Dict[str, str] = {
    #     "path": "other-data/data/WebKP 2",
    #     "mappings_folder_name": "mappings/pub/WebKP 2",
    #     "join_result_folder_path": "joined-df/pub/WebKP 2",
    #     "label_column": "class_label",
    #     "base_table_name": "webpage.csv",
    # }
    #
    # pub_repo = {
    #     'path': "other-data/data",
    #     'mappings_folder_name': "mappings/pub",
    #     'join_result_folder_path': "joined-df/pub",
    #     "label_column": "class_label",
    #     "base_table_name": "CiteSeer/paper.csv"
    # }
