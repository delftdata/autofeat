from neo4j import GraphDatabase

from feature_discovery.graph_processing.neo4j_queries import (
    _create_relation,
    _merge_nodes_relation_tables,
    _merge_nodes_relation,
    _drop_graph,
    _find_graph,
    _create_virtual_graph,
    _enumerate_all_paths,
    _get_relation_properties,
    _get_node_by_id,
    _get_node_by_source_name,
    _get_pk_fk_nodes,
    _get_adjacent_nodes,
    _get_relation_properties_node_name,
)

from feature_discovery.graph_processing.neo4j_queries import _get_adjacent_nodes_rels

driver = GraphDatabase.driver(
    f"neo4j://localhost:7687",
    auth=("neo4j", "pass1234")
    # f"neo4j://neo4j:7687", auth=("neo4j", "pass")
)


def create_relation(from_node_id, to_node_id, relation_name, weight=1):
    with driver.session() as session:
        relation = session.write_transaction(
            _create_relation, from_node_id, to_node_id, relation_name, weight
        )
    return relation


def merge_nodes_relation_tables(
    a_table_name, b_table_name, a_table_path, b_table_path, a_col, b_col, weight=1
):
    with driver.session() as session:
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


def merge_nodes_relation(a_name, b_name, table_name, table_path, rel_name, weight=1):
    with driver.session() as session:
        result = session.write_transaction(
            _merge_nodes_relation,
            a_name,
            b_name,
            table_name,
            table_path,
            rel_name,
            weight,
        )
    return result


def drop_graph(name):
    with driver.session() as session:
        session.write_transaction(_drop_graph, name)


def find_graph(name):
    with driver.session() as session:
        result = session.write_transaction(_find_graph, name)
    return result


def init_graph(name):
    with driver.session() as session:
        session.write_transaction(_create_virtual_graph, name)


def enumerate_all_paths(name):
    with driver.session() as session:
        result = session.write_transaction(_enumerate_all_paths, name)

    return result


def get_relation_properties(from_id, to_id):
    with driver.session() as session:
        result = session.write_transaction(_get_relation_properties, from_id, to_id)

    return result


def get_relation_properties_node_name(from_id, to_id):
    with driver.session() as session:
        result = session.write_transaction(
            _get_relation_properties_node_name, from_id, to_id
        )

    return result


def get_node_by_id(node_id):
    with driver.session() as session:
        result = session.write_transaction(_get_node_by_id, node_id)

    return result


def get_node_by_source_name(source_name):
    with driver.session() as session:
        result = session.write_transaction(_get_node_by_source_name, source_name)

    return result


def get_pk_fk_nodes(source_path):
    with driver.session() as session:
        result = session.write_transaction(_get_pk_fk_nodes, source_path)
    return result


def get_adjacent_nodes_rels(node_id) -> dict:
    """
    Computes a dictionary of adjacent columns and the corresponding edges.
    :param node_id: The ID of the node whose adjacent values to find
    :return: A dictionary of nodes and aggregated relationships.

    Example: {
                node1: [
                    {
                        from_column: col1, // the column of node_id table
                        weight: 0.8, // similarity score
                        to_column: col2 // the column of node1 table
                    },
                    {etc},
                    {etc}
                ]
            }
    """
    with driver.session() as session:
        result = session.write_transaction(_get_adjacent_nodes_rels, node_id)
    return result


def get_adjacent_nodes(node_id) -> list:
    """
    Computes a list of adjacent node IDs.
    :param node_id: The ID of the node whose adjacent values to find
    :return: A list of node IDs.
    """
    with driver.session() as session:
        result = session.write_transaction(_get_adjacent_nodes, node_id)
    return result
