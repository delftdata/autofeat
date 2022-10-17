from typing import List

from algorithms import TRAINING_FUNCTIONS
from data_preparation.dataset_base import Dataset
from data_preparation.utils import prepare_data_for_ml
from experiments.utils import hp_tune_join_all
from experiments.result_object import Result


class BaseTableExperiment:
    def __init__(self, data: Dataset):
        self.dataset = data
        self.dataset.set_features()
        self.results: List[Result] = []
        self.approach = Result.BASE

    def compute_accuracy_results(self):
        print(f'======== NON-AUG Pipeline ========')

        if self.dataset.base_table_df is None:
            self.dataset.set_base_table_df()

        X, y = prepare_data_for_ml(dataframe=self.dataset.base_table_df, target_column=self.dataset.target_column)

        for algorithm in TRAINING_FUNCTIONS:
            print(f"==== Model Name: {algorithm.LABEL} ====")
            accuracy, max_depth, feature_importances, train_time, _ = hp_tune_join_all(
                X, y, algorithm().train, False
            )
            entry = Result(
                approach=self.approach,
                data_path=self.dataset.base_table_id,
                data_label=self.dataset.base_table_label,
                algorithm=algorithm.LABEL,
                depth=max_depth,
                accuracy=accuracy,
                feature_importance=feature_importances,
                train_time=train_time,
            )
            self.results.append(entry)

        print(f"======== Finished NON-AUG Pipeline ========")
