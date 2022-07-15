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

    citeSeer_data: Dict[str, str] = {
        "path": "other-data/data/CiteSeer",
        "mappings_folder_name": "mappings/pub/CiteSeer",
        "join_result_folder_path": "joined-df/pub/CiteSeer",
        "label_column": "class_label",
        "base_table_name": "paper.csv",
    }

    cora_data: Dict[str, str] = {
        "path": "other-data/data/CORA",
        "mappings_folder_name": "mappings/pub/CORA",
        "join_result_folder_path": "joined-df/pub/CORA",
        "label_column": "class_label",
        "base_table_name": "paper.csv",
    }

    # pubmed_data: Dict[str, str] = {
    #     "path": "other-data/data/PubMed_Diabetes",
    #     "mappings_folder_name": "mappings/pub/PubMed_Diabetes",
    #     "join_result_folder_path": "joined-df/pub/PubMed_Diabetes",
    #     "label_column": "class_label",
    #     "base_table_name": "paper.csv",
    # }

    web_data: Dict[str, str] = {
        "path": "other-data/data/WebKP 2",
        "mappings_folder_name": "mappings/pub/WebKP 2",
        "join_result_folder_path": "joined-df/pub/WebKP 2",
        "label_column": "class_label",
        "base_table_name": "webpage.csv",
    }
