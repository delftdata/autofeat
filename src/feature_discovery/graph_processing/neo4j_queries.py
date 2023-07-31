def _merge_nodes_relation_tables(tx, a_table_name, b_table_name, a_table_path, b_table_path, a_col, b_col, weight):
    tx_result = tx.run(
        "merge (a:Node {id: $a_table_path, label: $a_table_name}) "
        "merge (b:Node {id: $b_table_path, label: $b_table_name}) "
        "merge (a)-[r:RELATED {from_column: $a_col, to_column: $b_col, from_label: $a_table_path, to_label: $b_table_path}]-(b) "
        "on create set r.weight = $weight "
        "on match set r.weight = case when r.weight < $weight then $weight else r.weight end",
        a_table_path=a_table_path,
        a_table_name=a_table_name,
        b_table_path=b_table_path,
        b_table_name=b_table_name,
        a_col=a_col,
        b_col=b_col,
        weight=weight,
    )

    record = tx_result.single()
    return record


def _get_relation_properties(tx, from_id, to_id):
    tx_result = tx.run(
        "match (n {id: $from_id})-[r:RELATED]-(m {id: $to_id}) return properties(r) as props",
        from_id=from_id,
        to_id=to_id,
    )
    values = []
    for record in tx_result:
        values.append(record["props"])
    return values


def _get_relation_properties_node_name(tx, from_id, to_id):
    tx_result = tx.run(
        "match (n {id: $from_id})-[r:RELATED]-(m {id: $to_id}) return properties(r) as props, n.id as from_label, m.id as to_label order by r.weight desc",
        from_id=from_id,
        to_id=to_id,
    )
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


def _get_pk_fk_nodes(tx, source_path):
    tx_result = tx.run(
        "match (n {id: $source_path})-[r:RELATED {weight: 1}]-(m) " "return n, m", source_path=str(source_path)
    )

    values = []
    for record in tx_result:
        values.append(record.values())
    return values


def _get_adjacent_nodes(tx, node_id):
    tx_result = tx.run(
        "match (n:Node {id: $node_id})-[r]-(m:Node) with r, m order by r.weight desc return distinct m.id as id",
        node_id=node_id
    )
    values = []
    for record in tx_result:
        values.append(record["id"])
    return values


def _export_all_connections(tx):
    tx_results = tx.run(
        "match (n)-[r]-(m) "
        "where n.id=r.from_label and m.id=r.to_label "
        "with n, m, r, "
        "split(n.id, n.label)[0] as from_path, "
        "split(m.id, m.label)[0] as to_path "
        "return from_path, n.label as from_table, r.from_column as from_column, "
        "to_path, m.label as to_label, r.to_column as to_column"
    )
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
        dataset_label=dataset_label,
    )
    return _parse_records(tx_results)


def _parse_records(tx_results):
    values = []
    for record in tx_results:
        values.append(
            {
                "from_path": record["from_path"],
                "from_table": record["from_table"],
                "from_column": record["from_column"],
                "to_path": record["to_path"],
                "to_label": record["to_label"],
                "to_column": record["to_column"],
            }
        )

    return values


def _create_node(tx, a_table_path, a_table_name):
    tx_result = tx.run(
        "MERGE (a:Node {id: $a_table_path, label: $a_table_name}) return a",
        a_table_path=a_table_path,
        a_table_name=a_table_name,
    )

    return tx_result.single()
