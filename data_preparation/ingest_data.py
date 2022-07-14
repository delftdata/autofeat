import csv
import glob
import itertools
import json
import os

import pandas as pd
from valentine import valentine_match
from valentine.algorithms import Coma

from utils.file_naming_convention import MAPPING, CONNECTIONS
from utils.neo4j_utils import create_table_node, create_relation_between_table_nodes, get_relation_between_table_nodes

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


def ingest_connections(directory_path: str, mapping: dict):
    files = glob.glob(f"../{directory_path}/**/{CONNECTIONS}", recursive=True)

    for f in files:
        print(f"Ingesting connections from {f}")
        with open(f) as rd:
            connections = list(csv.reader(rd))
        source = f.split('/')[-2]
        for link in connections:
            source_name, source_id, target_name, target_id = link
            create_relation_between_table_nodes(mapping[f"{source}/{source_name}"], mapping[f"{source}/{target_name}"],
                                                source_id, target_id)


def profile_valentine_all(directory_path: str):
    files = glob.glob(f"../{directory_path}/**/*.csv", recursive=True)
    files = [f for f in files if CONNECTIONS not in f]

    for table_pair in itertools.combinations(files, r=2):
        (tab1, tab2) = table_pair
        print(f"Processing the match between:\n\t{tab1}\n\t{tab2}")
        df1 = pd.read_csv(tab1, header=0, engine="python", encoding="utf8", quotechar='"', escapechar='\\')
        df2 = pd.read_csv(tab2, header=0, engine="python", encoding="utf8", quotechar='"', escapechar='\\')
        matches = valentine_match(df1, df2, Coma(strategy="COMA_OPT"))

        for ((_, col_from), (_, col_to)), similarity in matches.items():
            if similarity > threshold:
                print(f"Similarity {similarity} between:\n\t{tab1} -- {col_from}\n\t{tab2} -- {col_to}")
                relation = get_relation_between_table_nodes(tab1, tab2, col_from, col_to)
                if not relation or relation.get('weight') < similarity:
                    create_relation_between_table_nodes(tab1, tab2, col_from, col_to, similarity)


