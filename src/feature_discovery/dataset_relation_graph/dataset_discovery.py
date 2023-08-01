import glob
import itertools
from typing import List

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from valentine import valentine_match
from valentine.algorithms import Coma

from feature_discovery.config import DATA_FOLDER, CONNECTIONS
from feature_discovery.graph_processing.neo4j_transactions import merge_nodes_relation_tables


def profile_valentine_all(valentine_threshold: float = 0.55):
    files = glob.glob(f"{DATA_FOLDER}/**/*.csv", recursive=True)
    files = [f for f in files if CONNECTIONS not in f]

    profile_valentine_logic(files, valentine_threshold)


def profile_valentine_dataset(dataset_name: str, valentine_threshold: float = 0.55):
    files = glob.glob(f"{DATA_FOLDER / dataset_name}/**/*.csv", recursive=True)
    files = [f for f in files if CONNECTIONS not in f]

    profile_valentine_logic(files, valentine_threshold)


def profile_valentine_logic(files: List[str], valentine_threshold: float = 0.55):
    def profile(table_pair):
        (tab1, tab2) = table_pair

        a_table_path = tab1.partition(f"{DATA_FOLDER}/")[2]
        b_table_path = tab2.partition(f"{DATA_FOLDER}/")[2]

        a_table_name = a_table_path.split("/")[-1]
        b_table_name = b_table_path.split("/")[-1]

        print(f"Processing the match between:\n\t{a_table_path}\n\t{b_table_path}")
        df1 = pd.read_csv(tab1, encoding="utf8")
        df2 = pd.read_csv(tab2, encoding="utf8")
        matches = valentine_match(df1, df2, Coma(strategy="COMA_OPT"))

        for item in matches.items():
            ((_, col_from), (_, col_to)), similarity = item
            if similarity > valentine_threshold:
                print(f"Similarity {similarity} between:\n\t{a_table_path} -- {col_from}\n\t{b_table_path} -- {col_to}")

                merge_nodes_relation_tables(a_table_name=a_table_name,
                                            b_table_name=b_table_name,
                                            a_table_path=a_table_path,
                                            b_table_path=b_table_path,
                                            a_col=col_from,
                                            b_col=col_to,
                                            weight=similarity)

                merge_nodes_relation_tables(a_table_name=b_table_name,
                                            b_table_name=a_table_name,
                                            a_table_path=b_table_path,
                                            b_table_path=a_table_path,
                                            a_col=col_to,
                                            b_col=col_from,
                                            weight=similarity)

    Parallel(n_jobs=-1)(delayed(profile)(table_pair) for table_pair in tqdm(itertools.combinations(files, r=2)))
