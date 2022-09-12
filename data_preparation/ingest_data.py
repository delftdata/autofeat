import csv
import glob
import itertools
import json
import os

import pandas as pd
from valentine import valentine_match
from valentine.algorithms import Coma

from utils.file_naming_convention import MAPPING, CONNECTIONS, DATA_FOLDER, MAPPING_FOLDER
from utils.neo4j_utils import create_table_node, merge_nodes_relation, create_relation
from utils.relation_types import SIBLING, RELATED

folder_name = os.path.abspath(os.path.dirname(__file__))
threshold = 0.8


def ingest_fabricated_data() -> dict:
    files = glob.glob(f"../{DATA_FOLDER}/**/*.csv", recursive=True)
    # Filter out connections.csv file
    files = [f for f in files if CONNECTIONS not in f and f.endswith("csv")]
    mapping = {}
    for f in files:
        table_path = f
        table_name = f.partition(f"{DATA_FOLDER}/")[2]
        print(f"Creating nodes from {f}")
        df = pd.read_csv(f, header=0, engine="python", encoding="utf8", quotechar='"', escapechar='\\', nrows=1)
        for column_pair in itertools.combinations(df.columns, r=2):
            (col1, col2) = column_pair
            merge_nodes_relation(f"{table_path}/{col1}", f"{table_name}/{col1}", col1, table_path,
                                 f"{table_path}/{col2}", f"{table_name}/{col2}", col2, table_path, SIBLING, 0)

        mapping[table_name] = table_path

    return mapping


def ingest_connections():
    files = glob.glob(f"../{DATA_FOLDER}/**/{CONNECTIONS}", recursive=True)

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


def profile_valentine_all():
    files = glob.glob(f"../{DATA_FOLDER}/**/*.csv", recursive=True)
    files = [f for f in files if CONNECTIONS not in f]

    for table_pair in itertools.combinations(files, r=2):
        (tab1, tab2) = table_pair
        print(f"Processing the match between:\n\t{tab1}\n\t{tab2}")
        df1 = pd.read_csv(tab1, header=0, engine="python", encoding="utf8", quotechar='"', escapechar='\\')
        df2 = pd.read_csv(tab2, header=0, engine="python", encoding="utf8", quotechar='"', escapechar='\\')
        matches = valentine_match(df1, df2, Coma(strategy="COMA_OPT"))

        for item in matches.items():
            ((_, col_from), (_, col_to)), similarity = item
            if similarity > threshold:
                print(f"Similarity {similarity} between:\n\t{tab1} -- {col_from}\n\t{tab2} -- {col_to}")

                node_id_source = f"{tab1}/{col_from}"
                node_id_target = f"{tab2}/{col_to}"

                create_relation(node_id_source, node_id_target, RELATED, similarity)

