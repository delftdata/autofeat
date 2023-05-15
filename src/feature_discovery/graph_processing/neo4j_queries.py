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


def _merge_nodes_relation_tables(tx, a_table_name, b_table_name, a_table_path, b_table_path, a_col, b_col, weight):
    tx_result = tx.run("merge (a:Node {id: $a_table_path, label: $a_table_name}) "
                       "merge (b:Node {id: $b_table_path, label: $b_table_name}) "
                       "merge (a)-[r:RELATED {from_column: $a_col, to_column: $b_col, from_label: $a_table_name, to_label: $b_table_name}]-(b) "
                       "on create set r.weight = $weight "
                       "on match set r.weight = case when r.weight < $weight then $weight else r.weight end",
                       a_table_path=a_table_path, a_table_name=a_table_name,
                       b_table_path=b_table_path, b_table_name=b_table_name, a_col=a_col, b_col=b_col, weight=weight)

    record = tx_result.single()
    return record


def _merge_nodes_relation(tx, a_name, b_name, table_name, table_path, rel_name, weight):
    a_id = f"{table_path}/{a_name}"
    b_id = f"{table_path}/{b_name}"
    a_label = f"{table_name}/{a_name}"
    b_label = f"{table_name}/{b_name}"
    tx_result = tx.run(
        "merge (a:Node {id: $a_id, label: $a_label, name: $a_name, source_name: $source_name, source_path: $source_path}) "
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


def _drop_graph(tx, name):
    tx.run(f"CALL gds.graph.drop('{name}')")


def _find_graph(tx, name):
    tx_result = tx.run("CALL gds.graph.list($name)", name=name)
    values = []
    for record in tx_result:
        values.append(record.values())
    return values


def _create_virtual_graph(tx, name):
    result = tx.run("CALL gds.graph.project("
                    "$name, "
                    "'Node', "
                    "{ LINK: "
                    "{ type: 'RELATED', "
                    "orientation: 'UNDIRECTED'}})", name=name)


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


def _get_relation_properties(tx, from_id, to_id):
    tx_result = tx.run("match (n {id: $from_id})-[r:RELATED]-(m {id: $to_id}) return properties(r) as props",
                       from_id=from_id, to_id=to_id)
    values = []
    for record in tx_result:
        values.append(record["props"])
    return values


def _get_relation_properties_node_name(tx, from_id, to_id):
    tx_result = tx.run("match (n {id: $from_id})-[r:RELATED]-(m {id: $to_id}) "
                       "return properties(r) as props, n.label as from_label, m.label as to_label "
                       "order by r.weight desc", from_id=from_id, to_id=to_id)
    values = []
    for record in tx_result:
        values.append(record.values())
    return values


def _get_node_by_id(tx, node_id):
    result = tx.run("match (n {id: $node_id}) return n as node", node_id=str(node_id))
    record = result.single()
    if not record:
        return None
    return record['node']


def _get_node_by_source_name(tx, source_name):
    tx_result = tx.run("match (n {source_path: $source_name}) return n as node", source_name=str(source_name))
    result = []
    for record in tx_result:
        result.append(record['node'])
    return result


def _get_pk_fk_nodes(tx, source_path):
    tx_result = tx.run("match (n {id: $source_path})-[r:RELATED {weight: 1}]-(m) "
                       "return n, m", source_path=str(source_path))

    values = []
    for record in tx_result:
        values.append(record.values())
    return values


def _get_adjacent_nodes_rels(tx, node_id):
    tx_result = tx.run("match (n:Node {id: $node_id})-[r]-(m:Node) "
                       "return m.id as id, collect(properties(r)) as props", node_id=node_id)
    values = {}
    for record in tx_result:
        vals = record.values()
        values.update({vals[0]: vals[1]})
    return values


def _get_adjacent_nodes(tx, node_id):
    tx_result = tx.run("match (n:Node {id: $node_id})-[r]-(m:Node) "
                       "return distinct m.id as id order by id asc", node_id=node_id)
    values = []
    for record in tx_result:
        values.append(record["id"])
    return values


def _export_all_connections(tx):
    tx_results = tx.run(
        "match (n)-[r]-(m) "
        "where n.label=r.from_label and m.label=r.to_label "
        "with n, m, r, "
        "split(n.id, n.label)[0] as from_path, "
        "split(m.id, m.label)[0] as to_path "
        "return from_path, n.label as from_table, r.from_column as from_column, "
        "to_path, m.label as to_label, r.to_column as to_column")
    return _parse_records(tx_results)


def _export_dataset_connections(tx, dataset_label):
    tx_results = tx.run(
        "match (n)-[r]-(m) "
        "where n.id contains $dataset_label and n.label=r.from_label and m.label=r.to_label "
        "with n, m, r, "
        "split(n.id, n.label)[0] as from_path, "
        "split(m.id, m.label)[0] as to_path "
        "return from_path, n.label as from_table, r.from_column as from_column, "
        "to_path, m.label as to_label, r.to_column as to_column",
        dataset_label=dataset_label)
    return _parse_records(tx_results)


def _parse_records(tx_results):
    values = []
    for record in tx_results:
        values.append({
            "from_path": record["from_path"],
            "from_table": record["from_table"],
            "from_column": record["from_column"],
            "to_path": record["to_path"],
            "to_label": record["to_label"],
            "to_column": record["to_column"]
        })

    return values
