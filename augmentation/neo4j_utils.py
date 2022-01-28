from neo4j import GraphDatabase


driver = GraphDatabase.driver(
    f"neo4j://localhost:11005", auth=("neo4j", "pass")
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
