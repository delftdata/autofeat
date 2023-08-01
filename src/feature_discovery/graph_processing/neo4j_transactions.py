from typing import List

from neo4j import GraphDatabase

from feature_discovery.config import NEO4J_HOST, NEO4J_CREDENTIALS, NEO4J_DATABASE
from feature_discovery.graph_processing.neo4j_queries import (
    _merge_nodes_relation_tables,
    _get_relation_properties,
    _get_node_by_id,
    _get_pk_fk_nodes,
    _get_adjacent_nodes,
    _get_relation_properties_node_name,
    _export_all_connections,
    _export_dataset_connections, _create_node,
)

driver = GraphDatabase.driver(NEO4J_HOST, auth=NEO4J_CREDENTIALS)


def merge_nodes_relation_tables(a_table_name, b_table_name, a_table_path, b_table_path, a_col, b_col, weight=1):
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.write_transaction(
            _merge_nodes_relation_tables,
            a_table_name,
            b_table_name,
            a_table_path,
            b_table_path,
            a_col,
            b_col,
            weight,
        )
    return result


def get_relation_properties(from_id, to_id):
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.write_transaction(_get_relation_properties, from_id, to_id)

    return result


def get_relation_properties_node_name(from_id, to_id):
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.write_transaction(_get_relation_properties_node_name, from_id, to_id)

    return result


def get_node_by_id(node_id):
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.write_transaction(_get_node_by_id, node_id)

    return result


def get_pk_fk_nodes(source_path):
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.write_transaction(_get_pk_fk_nodes, source_path)
    return result


def get_adjacent_nodes(node_id) -> list:
    """
    Computes a list of adjacent node IDs.
    :param node_id: The ID of the node whose adjacent values to find
    :return: A list of node IDs.
    """
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.write_transaction(_get_adjacent_nodes, node_id)
    return result


def export_all_connections() -> List[dict]:
    with driver.session(database=NEO4J_DATABASE) as session:
        results = session.write_transaction(_export_all_connections)
    return results


def export_dataset_connections(dataset_label: str) -> List[dict]:
    with driver.session(database=NEO4J_DATABASE) as session:
        results = session.write_transaction(_export_dataset_connections, dataset_label)
    return results


def create_node(node_id, node_label):
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.write_transaction(_create_node, node_id, node_label)
    return result