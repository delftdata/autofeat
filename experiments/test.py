from augmentation.trial_error import traverse_join_pipeline
from graph_processing.traverse_graph import dfs_traversal

node_id = "/Users/andra/Developer/auto-data-augmentation/data/ARDA/school/base.csv"
target = "class"
base_table_features = ["DBN", "School Name", "School Type", "Total Parent Response Rate (%)",
                       "Total Teacher Response Rate (%)", "Total Student Response Rate (%)"]


def test_arda():
    from arda.arda import select_arda_features_budget_join

    X, y, selected_features, join_time, fs_time = select_arda_features_budget_join(node_id, target, base_table_features,
                                                                                   sample_size=1000)
    print(f"X shape: {X.shape}\nSelected features:\n\t{selected_features}")


def test_pipeline():
    visited = []
    join_path_tree = {}
    train_results = []
    dfs_traversal(base_node_id=node_id, discovered=visited, join_tree=join_path_tree)
    traverse_join_pipeline(base_node_id=node_id, target_column=target,
                           join_tree=join_path_tree, train_results=train_results)


test_pipeline()
