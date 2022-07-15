from typing import Dict


class Datasets:

    titanic_data: Dict[str, str] = {
        "join_result_folder_path": "joined-df/titanic",
        "label_column": "Survived",
        "base_table_name": "titanic.csv",
        "path": "other-data/decision-trees-split/titanic",
        "mappings_folder_name": "mappings/titanic",
    }

    steel_data: Dict[str, str] = {
        "join_result_folder_path": "joined-df/steel-plate-fault",
        "label_column": "Class",
        "base_table_name": "steel_plate_fault.csv",
        "path": "other-data/decision-trees-split/steel-plate-fault",
        "mappings_folder_name": "mappings/steel-plate-fault",
    }

    football_data: Dict[str, str] = {
        "join_result_folder_path": "joined-df/football",
        "label_column": "win",
        "base_table_name": "football.csv",
        "path": "other-data/decision-trees-split/football",
        "mappings_folder_name": "mappings/football",
    }

    # kidney_data: Dict[str, str] = {
    #     "join_result_folder_path": "joined-df/kidney-disease",
    #     "label_column": "classification",
    #     "base_table_name": "kidney_disease.csv",
    #     "path": "other-data/decision-trees-split/kidney-disease",
    #     "mappings_folder_name": "mappings/kidney-disease",
    # }

    pub_data: Dict[str, str] = {
        "path": "data",
        "mappings_folder_name": "mappings/pub",
        "join_result_folder_path": "joined-df/pub",
        "label_column": "class_label",
        "base_table_name": "PubMed_Diabetes/paper.csv",
    }
