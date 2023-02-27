from arda.arda import select_arda_features_budget_join
from data_preparation.join_data import join_tables
from graph_processing.traverse_graph import dfs_traversal

node_id = "/Users/andra/Developer/auto-data-augmentation/data/ARDA/school/base.csv"
target = "class"
base_table_features = ["DBN", "School Name", "School Type", "Total Parent Response Rate (%)",
                       "Total Teacher Response Rate (%)", "Total Student Response Rate (%)"]


def test_arda():
    # X, y, selected_features = select_arda_features_budget_join(node_id, target, base_table_features, sample_size=1000)
    # print(f"X shape: {X.shape}\nSelected features:\n\t{selected_features}")


if __name__ == "__main__":
    visited = []
    join_path_tree = {}
    dfs_traversal(base_node_id=node_id, discovered=visited, join_tree=join_path_tree)
    join_tables(base_node_id=node_id, target_column=target, join_path_list=visited, join_tree=join_path_tree)


