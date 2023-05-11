import csv
import glob
import itertools
import os
from typing import List

import pandas as pd
from valentine import valentine_match
from valentine.algorithms import Coma

from config import CONNECTIONS, DATA_FOLDER, VALENTINE_CONNECTIONS
from data_preparation import SIBLING, RELATED
from graph_processing.neo4j_transactions import merge_nodes_relation, create_relation, merge_nodes_relation_tables


def ingest_fabricated_data() -> dict:
    files = glob.glob(f"{DATA_FOLDER}/**/*.csv", recursive=True)
    # Filter out connections.csv file
    files = [f for f in files if CONNECTIONS not in f and f.endswith("csv")]
    mapping = {}
    for f in files:
        table_path = f
        table_name = f.partition(f"{DATA_FOLDER}/")[2]
        print(f"Creating nodes from {f}")
        df = pd.read_csv(f, encoding="utf8", nrows=1)
        for column_pair in itertools.combinations(df.columns, r=2):
            (col1, col2) = column_pair
            merge_nodes_relation(col1, col2, table_name, table_path, SIBLING, 0)

        mapping[table_name] = table_path

    return mapping


def ingest_unprocessed_data(dataset_name: str):
    print("Process the tables ...")
    files = glob.glob(f"{DATA_FOLDER / dataset_name}/**/*.csv", recursive=True)
    # Filter out connections.csv file
    files = [f for f in files if CONNECTIONS not in f and f.endswith("csv")]
    mapping = {}

    for f in files:
        table_path = f
        table_name = f.partition(f"{DATA_FOLDER / dataset_name}/")[2]

        mapping[table_name] = table_path

    print("Add the ground-truth ... ")
    connection_filename = glob.glob(f"{DATA_FOLDER / dataset_name}/**/{CONNECTIONS}", recursive=True)[0]
    connections = pd.read_csv(connection_filename)

    for index, row in connections.iterrows():
        merge_nodes_relation_tables(a_table_name=row["fk_table"], a_table_path=mapping[row["fk_table"]],
                                    b_table_name=row["pk_table"], b_table_path=mapping[row["pk_table"]],
                                    a_col=row["fk_column"], b_col=row["pk_column"], weight=1)
        merge_nodes_relation_tables(a_table_name=row["pk_table"], a_table_path=mapping[row["pk_table"]],
                                    b_table_name=row["fk_table"], b_table_path=mapping[row["fk_table"]],
                                    a_col=row["pk_column"], b_col=row["fk_column"], weight=1)

    return mapping


def ingest_tables() -> dict:
    files = glob.glob(f"{DATA_FOLDER}/**/{VALENTINE_CONNECTIONS}", recursive=True)
    connections = pd.read_csv(files[0])

    mapping = {}
    for index, row in connections.iterrows():
        merge_nodes_relation_tables(a_table_name=row["a_label"], a_table_path=row["a_id"],
                                    b_table_name=row["b_label"], b_table_path=row["b_id"],
                                    a_col=row["a_col"], b_col=row["b_col"], weight=row["weight"])
        mapping[row["a_label"]] = row["a_id"]
        mapping[row["b_label"]] = row["b_id"]

    return mapping


def ingest_connections():
    files = glob.glob(f"{DATA_FOLDER}/**/{CONNECTIONS}", recursive=True)

    for f in files:
        print(f"Ingesting connections from {f}")
        with open(f) as rd:
            connections = list(csv.reader(rd))
        node_source_name = f.partition(CONNECTIONS)[0]

        for link in connections[1:]:
            source_name, source_id, target_name, target_id = link
            node_id_source = f"{node_source_name}{source_name}/{source_id}"
            node_id_target = f"{node_source_name}{target_name}/{target_id}"

            create_relation(node_id_source, node_id_target, RELATED)


def profile_valentine_all(valentine_threshold: float = 0.8):
    files = glob.glob(f"{DATA_FOLDER}/**/*.csv", recursive=True)
    files = [f for f in files if CONNECTIONS not in f]

    profile_valentine_logic(files, valentine_threshold)


def profile_valentine_dataset(dataset_name: str, valentine_threshold: float = 0.8):
    files = glob.glob(f"{DATA_FOLDER / dataset_name}/**/*.csv", recursive=True)
    files = [f for f in files if CONNECTIONS not in f]

    profile_valentine_logic(files, valentine_threshold)


def profile_valentine_logic(files: List[str], valentine_threshold: float = 0.8):
    for table_pair in itertools.combinations(files, r=2):
        (tab1, tab2) = table_pair
        a_table_name = tab1.split("/")[-1]
        b_table_name = tab2.split("/")[-1]
        print(f"Processing the match between:\n\t{tab1}\n\t{tab2}")
        df1 = pd.read_csv(tab1, encoding="utf8")
        df2 = pd.read_csv(tab2, encoding="utf8")
        matches = valentine_match(df1, df2, Coma(strategy="COMA_OPT"))

        for item in matches.items():
            ((_, col_from), (_, col_to)), similarity = item
            if similarity > valentine_threshold:
                print(f"Similarity {similarity} between:\n\t{tab1} -- {col_from}\n\t{tab2} -- {col_to}")

                merge_nodes_relation_tables(a_table_name=a_table_name,
                                            b_table_name=b_table_name,
                                            a_table_path=tab1,
                                            b_table_path=tab2,
                                            a_col=col_from,
                                            b_col=col_to,
                                            weight=similarity)

