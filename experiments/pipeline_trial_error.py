from data_preparation.join_data import join_tables
from graph_processing.traverse_graph import dfs_traversal

node_id = "/Users/andra/Developer/auto-data-augmentation/data/ARDA/school/base.csv"
target = "class"
base_table_features = ["DBN", "School Name", "School Type", "Total Parent Response Rate (%)",
                       "Total Teacher Response Rate (%)", "Total Student Response Rate (%)"]


def pipeline():
    visited = []
    join_path_tree = {}
    dfs_traversal(base_node_id=node_id, discovered=visited, join_tree=join_path_tree)
    join_tables(base_node_id=node_id, target_column=target, join_path_list=visited, join_tree=join_path_tree)



if __name__ == "__main__":
    pipeline()


