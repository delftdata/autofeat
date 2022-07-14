import csv
import glob
import itertools
import json
import os

import pandas as pd
from valentine import valentine_match
from valentine.algorithms import Coma

from utils.file_naming_convention import MAPPING, CONNECTIONS
from utils.neo4j_utils import create_table_node, merge_nodes_relation

folder_name = os.path.abspath(os.path.dirname(__file__))
threshold = 0.7


def ingest_fabricated_data(directory_path: str, mappings_path) -> dict:
    files = glob.glob(f"../{directory_path}/**/*.csv", recursive=True)
    mapping = {}
    for f in files:
        table_path = f
        table_name = '/'.join(f.split('/')[-2:])

        if CONNECTIONS not in f and f.endswith("csv"):
            print(f"Creating nodes from {f}")
            mapping[table_name] = table_path
            node = create_table_node(table_name, table_path)

    with open(f"{os.path.join(folder_name, '../', mappings_path)}/{MAPPING}", 'w') as fp:
        json.dump(mapping, fp)
    return mapping


def ingest_connections(directory_path: str, mappings_path=None):
    files = glob.glob(f"../{directory_path}/**/{CONNECTIONS}", recursive=True)
    mapping = {}

    for f in files:
        print(f"Ingesting connections from {f}")
        with open(f) as rd:
            connections = list(csv.reader(rd))
        node_source_name = f.partition(CONNECTIONS)[0]
        source_folder = node_source_name.split('/')[-2]

        for link in connections:
            source_name, source_id, target_name, target_id = link
            node_id_source = f"{node_source_name}{source_name}/{source_id}"
            node_id_target = f"{node_source_name}{target_name}/{target_id}"

            label_source = f"{source_folder}/{source_name}/{source_id}"
            label_target = f"{source_folder}/{target_name}/{target_id}"

            mapping[label_source] = f"{node_source_name}{source_name}"
            mapping[label_target] = f"{node_source_name}{target_name}"

            merge_nodes_relation(node_id_source, label_source, f"{source_folder}/{source_name}",
                                 node_id_target, label_target, f"{source_folder}/{target_name}")

            # create_relation_between_table_nodes(mapping[f"{source}/{source_name}"], mapping[f"{source}/{target_name}"],
            #                                     source_id, target_id)
    if mappings_path:
        with open(f"{os.path.join(folder_name, '../', mappings_path)}/{MAPPING}", 'w') as fp:
            json.dump(mapping, fp)
    return mapping


def profile_valentine_all(directory_path: str):
    files = glob.glob(f"../{directory_path}/**/*.csv", recursive=True)
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
                label_1 = '/'.join(tab1.split('/')[-2:])
                label_2 = '/'.join(tab2.split('/')[-2:])
                relation = merge_nodes_relation(f"{tab1}/{col_from}", f"{label_1}/{col_from}", label_1,
                                                f"{tab2}/{col_to}", f"{label_2}/{col_to}", label_2)
