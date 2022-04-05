from neo4j import GraphDatabase

from utils import relation_types

driver = GraphDatabase.driver(
    f"neo4j://localhost:7687", auth=("neo4j", "pass")
)


def _get_nodes_with_pk_from_table(tx, table_name: str) -> list:
    result = tx.run(
        "match (n) where n.source_name contains $table_name "
        "with collect(n.id) as ids "
        "match (m)-[r:pk_fk]-(o) where m.id in ids "
        "return m.source_name, [m.name, o.source_name, o.name]", table_name=table_name
    )
    values = []
    for record in result:
        values.append(record.values())
    return values


def get_nodes_with_pk_from_table(table_name: str) -> list:
    with driver.session() as session:
        result = session.write_transaction(_get_nodes_with_pk_from_table, table_name)
    return result


def create_node(source, source_path, label):
    with driver.session() as session:
        node = session.write_transaction(_create_node, source, source_path, label)
    return node


def create_relation(from_node_id, to_node_id, relation_name):
    with driver.session() as session:
        relation = session.write_transaction(_create_relation, from_node_id, to_node_id, relation_name)
    return relation


def create_relation_with_path(from_node_id, to_node_id, path_name_a, path_name_b, relation_name):
    with driver.session() as session:
        relation = session.write_transaction(_create_relation_with_path, from_node_id, to_node_id, path_name_a, path_name_b,
                                             relation_name)
    return relation


def _create_node(tx, source, source_path, label):
    tx_result = tx.run("CREATE (n:Node) "
                       "SET n.id = $source + '/' + $label, "
                       "n.name = $label, "
                       "n.source_name = $source, "
                       "n.source_path = $source_path "
                       "RETURN n as node", source=source, label=label, source_path=source_path)
    result = []
    for record in tx_result:
        result.append(record['node'])
    return result


def create_subsumption_relation(source):
    with driver.session() as session:
        relation = session.write_transaction(_create_subsumption_relation, source)


def set_relation_properties(a_id, b_id, a_path_name, b_path_name, relation_name, **kwargs):
    with driver.session() as session:
        result = session.write_transaction(_set_properties, a_id, b_id, a_path_name, b_path_name, relation_name, **kwargs)


def _create_subsumption_relation(tx, source):
    tx_result = tx.run("MATCH (a:Node), (b:Node) "
                       "WHERE a.source_name = $source AND b.source_name = $source AND NOT(a.id = b.id) "
                       f"MERGE (a)-[s:{relation_types.SIBLING}]-(b) "
                       "RETURN type(s) as relation", source=source)
    result = []
    for record in tx_result:
        result.append(record['relation'])
    return result


def _create_relation(tx, a_id, b_id, relation_name):
    tx_result = tx.run("MATCH (a:Node {id: $a_id}) WITH a "
                       "MATCH (b:Node {id: $b_id}) "
                       f"MERGE (a)-[r:{relation_name}]->(b) "
                       "RETURN r as relation", a_id=a_id, b_id=b_id)

    result = []
    for record in tx_result:
        result.append(record['relation'])
    return result


def _create_relation_with_path(tx, a_id, b_id, path_name_a, path_name_b, relation_name):
    tx_result = tx.run("MATCH (a:Node {id: $a_id, source_path: $path_name_a}) WITH a "
                       "MATCH (b:Node {id: $b_id, source_path: $path_name_b}) "
                       f"MERGE (a)-[r:{relation_name}]->(b) "
                       "RETURN r as relation", a_id=a_id, b_id=b_id, path_name_a=path_name_a, path_name_b=path_name_b)

    result = []
    for record in tx_result:
        result.append(record['relation'])
    return result


def _set_properties(tx, a_id, b_id, path_name_a, path_name_b, relation_name, **kwargs):
    set_query = 'SET '
    for i, key in enumerate(kwargs.keys()):
        set_query += 'r.{} = ${} '.format(key, key)
        if i < len(kwargs.keys()) - 1:
            set_query += ', '

    tx_result = tx.run("MATCH (a:Node)-[r:{}]->(b:Node) WHERE a.id = $a_id and a.source_path = $path_name_a and "
                       "b.id = $b_id and b.source_path = $path_name_b {} RETURN r as relation"
                       .format(relation_name, set_query), a_id=a_id, b_id=b_id, path_name_a=path_name_a,
                       path_name_b=path_name_b, **kwargs)
    result = []
    for record in tx_result:
        result.append(record['relation'])
    return result
