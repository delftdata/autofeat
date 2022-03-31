import os

import pandas as pd
from numpy import genfromtxt

from utils.neo4j_utils import create_node, create_subsumption_relation, create_relation
from utils.relation_types import PK_FK

folder_name = os.path.abspath(os.path.dirname(__file__))


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
