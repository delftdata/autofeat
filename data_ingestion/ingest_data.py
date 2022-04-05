import itertools
import os

import pandas as pd
from numpy import genfromtxt
from valentine import valentine_match
from valentine.algorithms import Coma

from utils import relation_types
from utils.neo4j_utils import create_node, create_subsumption_relation, create_relation, create_relation_with_path, \
    set_relation_properties
from utils.relation_types import PK_FK

folder_name = os.path.abspath(os.path.dirname(__file__))
threshold = 0.7


def ingest_fabricated_data(directory_path: str):
    path = os.path.join(folder_name, "../", directory_path)
    for f in os.listdir(path):
        file_path = f"{path}/{f}"
        table_path = f"{directory_path}/{f}"
        table_name = f
        if "connections" not in f and f.endswith("csv"):
            print(f)
            df = pd.read_csv(file_path, header=0, engine="python", encoding="utf8", quotechar='"',
                             escapechar='\\')
            for col in df.columns:
                node = create_node(table_name, table_path, col)

            result = create_subsumption_relation(table_name)


def ingest_connections(directory_path: str):
    path = os.path.join(folder_name, "../", directory_path, "connections.csv")

    connections = list(map(lambda x: ((x[0], x[1]), (x[2], x[3])), genfromtxt(path, delimiter=',', dtype='str')))

    for link in connections:
        print(link)
        ((source_name, source_id), (target_name, target_id)) = link
        create_relation(f"{source_name}/{source_id}", f"{target_name}/{target_id}", PK_FK)


def profile_valentine_all(directory_path):
    path = os.path.join(folder_name, "../", directory_path)
    all_files = list(filter(lambda x: "connections" not in x and x.endswith("csv"), os.listdir(path)))

    for table_pair in itertools.combinations(all_files, r=2):
        (tab1, tab2) = table_pair
        df1 = pd.read_csv(f"{path}/{tab1}", header=0, engine="python", encoding="utf8", quotechar='"', escapechar='\\')
        df2 = pd.read_csv(f"{path}/{tab2}", header=0, engine="python", encoding="utf8", quotechar='"', escapechar='\\')
        matches = valentine_match(df1, df2, Coma(strategy="COMA_OPT"))

        for ((_, col_from), (_, col_to)), similarity in matches.items():
            if similarity > threshold:
                print(f"{tab1}/{col_from}")
                print(f"{tab2}/{col_to}")
                create_relation_with_path(f"{tab1}/{col_from}", f"{tab2}/{col_to}", f"{directory_path}/{tab1}",
                                          f"{directory_path}/{tab2}", relation_types.MATCH)
                set_relation_properties(f"{tab1}/{col_from}", f"{tab2}/{col_to}", f"{directory_path}/{tab1}",
                                        f"{directory_path}/{tab2}", relation_types.MATCH,
                                        coma=similarity)
