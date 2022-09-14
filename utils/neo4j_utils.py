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


def create_table_node(table_name, table_path):
    with driver.session() as session:
        node = session.write_transaction(_create_table_node, table_name, table_path)
    return node


def create_node(source, source_path, label):
    with driver.session() as session:
        node = session.write_transaction(_create_node, source, source_path, label)
    return node


def create_relation(from_node_id, to_node_id, relation_name, weight=1):
    with driver.session() as session:
        relation = session.write_transaction(_create_relation, from_node_id, to_node_id, relation_name, weight)
    return relation


def create_relation_with_path(from_node_id, to_node_id, path_name_a, path_name_b, relation_name):
    with driver.session() as session:
        relation = session.write_transaction(_create_relation_with_path, from_node_id, to_node_id, path_name_a,
                                             path_name_b,
                                             relation_name)
    return relation


def merge_nodes_relation(a_name, b_name, table_name, table_path, rel_name, weight=1):
    with driver.session() as session:
        result = session.write_transaction(_merge_nodes_relation, a_name, b_name, table_name, table_path, rel_name,
                                           weight)
    return result


def _merge_nodes_relation(tx, a_name, b_name, table_name, table_path, rel_name, weight):
    a_id = f"{table_path}/{a_name}"
    b_id = f"{table_path}/{b_name}"
    a_label = f"{table_name}/{a_name}"
    b_label = f"{table_name}/{b_name}"
    tx_result = tx.run("merge (a:Node {id: $a_id, label: $a_label, name: $a_name, source_name: $source_name, source_path: $source_path}) "
                       "merge (b:Node {id: $b_id, label: $b_label, name: $b_name, source_name: $source_name, source_path: $source_path}) "
                       f"merge (a)-[r:{rel_name}]-(b) "
                       "on match set (case when r.weight < $weight then r end).weight = $weight "
                       "on create set r.weight = $weight "
                       "return r as relation", a_id=a_id, b_id=b_id, a_label=a_label, b_label=b_label, a_name=a_name,
                       b_name=b_name, source_name=table_name, source_path=table_path, weight=weight)

    record = tx_result.single()
    if not record:
        return None
    return record['relation']


def _create_table_node(tx, table_name, table_path):
    tx_result = tx.run("MERGE (n:Node {id: $table_path}) "
                       "ON CREATE "
                       "SET n.label = $table_name "
                       "RETURN n as node", table_name=table_name, table_path=table_path)

    result = tx_result.single()
    return result["node"]


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
        result = session.write_transaction(_set_properties, a_id, b_id, a_path_name, b_path_name, relation_name,
                                           **kwargs)


def create_relation_between_table_nodes(from_id, to_id, from_key, to_key, weight=1):
    with driver.session() as session:
        result = session.write_transaction(_create_relation_between_table_nodes, from_id, to_id,
                                           from_key, to_key, weight)
    return result


def get_relation_between_table_nodes(from_id, to_id):
    with driver.session() as session:
        result = session.write_transaction(_get_relation_between_table_nodes, from_id, to_id)
    return result


def _create_subsumption_relation(tx, source):
    tx_result = tx.run("MATCH (a:Node), (b:Node) "
                       "WHERE a.source_name = $source AND b.source_name = $source AND NOT(a.id = b.id) "
                       f"MERGE (a)-[s:{relation_types.SIBLING}]-(b) "
                       "RETURN type(s) as relation", source=source)
    result = []
    for record in tx_result:
        result.append(record['relation'])
    return result


def _create_relation(tx, a_id, b_id, relation_name, weight):
    tx_result = tx.run("MATCH (a:Node {id: $a_id}) WITH a "
                       "MATCH (b:Node {id: $b_id}) "
                       f"MERGE (a)-[r:{relation_name}]-(b) "
                       "on match set (case when r.weight < $weight then r end).weight = $weight "
                       "on create set r.weight = $weight "
                       "RETURN r as relation", a_id=a_id, b_id=b_id, weight=weight)

    result = []
    for record in tx_result:
        result.append(record['relation'])
    return result


def _create_relation_between_table_nodes(tx, a_id, b_id, from_key, to_key, weight):
    tx_result = tx.run("MATCH (a:Node {id: $a_id}) WITH a "
                       "MATCH (b:Node {id: $b_id}) "
                       "MERGE (a)-[r:RELATED {from_key: $from_key, to_key: $to_key, weight: $weight}]->(b) "
                       "RETURN r as relation", a_id=a_id, b_id=b_id,
                       from_key=from_key, to_key=to_key, weight=weight)

    record = tx_result.single()
    return record['relation']


def _get_relation_between_table_nodes(tx, a_id, b_id):
    tx_result = tx.run(
        "MATCH (a:Node {id: $a_id})-[r:RELATED]->(b:Node {id: $b_id}) "
        "return r as relation", a_id=a_id, b_id=b_id)
    record = tx_result.single()
    if not record:
        return None
    return record['relation']


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


def _create_virtual_graph(tx, name):
    result = tx.run("CALL gds.graph.create("
                    "$name, "
                    "'Node', "
                    "{ LINK: "
                    "{ type: 'RELATED', "
                    "orientation: 'UNDIRECTED'}})", name=name)


def _drop_graph(tx, name):
    tx.run(f"CALL gds.graph.drop('{name}')")


def drop_graph(name):
    with driver.session() as session:
        session.write_transaction(_drop_graph, name)


def init_graph(name):
    with driver.session() as session:
        session.write_transaction(_create_virtual_graph, name)


def _enumerate_all_paths(tx, name):
    tx_result = tx.run(f"CALL gds.alpha.allShortestPaths.stream('{name}') "
                       "YIELD sourceNodeId, targetNodeId, distance "
                       "WITH sourceNodeId, targetNodeId, distance "
                       "WHERE distance=1 "
                       "MATCH (source:Node) WHERE id(source) = sourceNodeId "
                       "MATCH (target:Node) WHERE id(target) = targetNodeId "
                       "WITH source, target, distance WHERE source <> target "
                       "RETURN source.id AS source, target.id AS target, distance "
                       "ORDER BY distance ASC, source ASC, target ASC")
    values = []
    for record in tx_result:
        values.append(record.values())
    return values


def enumerate_all_paths(name):
    with driver.session() as session:
        result = session.write_transaction(_enumerate_all_paths, name)

    return result


def _get_relation_properties(tx, from_id, to_id):
    result = tx.run("match (n {id: $from_id})-[r:RELATED]-(m {id: $to_id}) return r.from_key, r.to_key",
                    from_id=from_id, to_id=to_id)
    return result.values()[0]


def get_relation_properties(from_id, to_id):
    with driver.session() as session:
        result = session.write_transaction(_get_relation_properties, from_id, to_id)

    return result


def get_node_by_id(node_id):
    with driver.session() as session:
        result = session.write_transaction(_get_node_by_id, node_id)

    return result


def _get_node_by_id(tx, node_id):
    result = tx.run("match (n {id: $node_id}) return n as node", node_id=node_id)
    record = result.single()
    if not record:
        return None
    return record['node']


def get_node_by_source_name(source_name):
    with driver.session() as session:
        result = session.write_transaction(_get_node_by_source_name, source_name)

    return result


def _get_node_by_source_name(tx, source_name):
    tx_result = tx.run("match (n {source_path: $source_name}) return n as node", source_name=source_name)
    result = []
    for record in tx_result:
        result.append(record['node'])
    return result
