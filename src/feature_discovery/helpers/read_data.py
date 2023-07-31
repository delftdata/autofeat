import pandas as pd

from feature_discovery.config import DATA_FOLDER
from feature_discovery.graph_processing.neo4j_transactions import get_node_by_id


def get_df_with_prefix(node_id: str, target_column=None) -> tuple:
    """
    Get the node from the database, read the file identified by node_id and prefix the column names with the node label.

    :param node_id: ID of the node - used to retrieve the corresponding node from the database
    :param target_column: Optional parameter. The name of the label/target column containing the classes,
            only needed when the dataset to read contains the class.
    :return: 0: A pandas dataframe whose columns are prefixed with the node label, 1: the node label
    """

    node_label = get_node_by_id(node_id).get("id")
    dataframe = pd.read_csv(str(DATA_FOLDER / node_id), header=0, engine="python", encoding="utf8", quotechar='"', escapechar='\\')
    if target_column:
        dataframe = dataframe.set_index([target_column]).add_prefix(f"{node_label}.").reset_index()
    else:
        dataframe = dataframe.add_prefix(f"{node_label}.")

    return dataframe, node_label
