from typing import List, Dict

from graph_processing.neo4j_transactions import get_adjacent_nodes


def dfs_traversal(base_node_id: str, discovered: List, join_tree: Dict):
    """
    The method traverses the graph and populates a list with the traversal path and a join tree.

    :param base_node_id: The ID of the starting node for the DFS traversal.
    :param discovered: Empty list. The list will be populated with the nodes representing the traversal path.
    :param join_tree: Empty dictionary. The dictionary will be populated with the nodes which create a join tree.
    The join tree shows how to join the tables (key - left part of the join, value - right part of the join).
    """
    if base_node_id not in discovered:
        discovered.append(base_node_id)
        adjacent_nodes = get_adjacent_nodes(base_node_id)

        if base_node_id not in join_tree.keys():
            join_tree[base_node_id] = {}

        for node in adjacent_nodes:
            if node not in discovered:
                # Create a tree structure such that we know how to join the tables
                join_tree[base_node_id].update({node: {}})
            dfs_traversal(node, discovered, join_tree[base_node_id])


if __name__ == "__main__":
    node_id = "/Users/andra/Developer/auto-data-augmentation/data/ARDA/school/base.csv"
    visited = []
    join_path_tree = {}
    dfs_traversal(base_node_id=node_id, discovered=visited, join_tree=join_path_tree)

    print(visited)
    print(join_path_tree)
