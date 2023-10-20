from feature_discovery.baselines.join_all import JoinAll
from feature_discovery.experiments.dataset_object import Dataset
from feature_discovery.experiments.init_datasets import init_datasets
from feature_discovery.experiments.utils_dataset import filter_datasets


def join_all_bfs(dataset: Dataset):
    joinall = JoinAll(
        base_table_id=str(dataset.base_table_id),
        target_column=dataset.target_column,
    )
    dataframe = joinall.join_all_bfs(queue={str(dataset.base_table_id)})
    dataframe.drop(columns=joinall.join_keys[joinall.partial_join_name], inplace=True)

    print("End")


if __name__ == "__main__":
    init_datasets()
    dataset = filter_datasets(["credit"])[0]
    join_all_bfs(dataset)
