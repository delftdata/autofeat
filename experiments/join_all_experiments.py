import time
from typing import List

from data_preparation.dataset_base import Dataset
from data_preparation.join_data import join_all
from data_preparation.utils import prepare_data_for_ml
from experiments.utils import hp_tune_join_all
from experiments.result_object import Result
from experiments.utils import TRAINING_FUNCTIONS, ID3


class JoinAllExperiment:
    def __init__(self, data: Dataset, do_feature_selection=False):
        self.dataset = data
        self.results: List[Result] = []
        self.approach = Result.JOIN_ALL_FS if do_feature_selection else Result.JOIN_ALL
        self.do_feature_selection = do_feature_selection

    def compute_accuracy_results(self):
        print(f'======== JOIN-ALL Dataset Pipeline - feature selection - {self.do_feature_selection} ========')
        start = time.time()
        dataset_df = join_all(self.dataset.base_table_id)
        end = time.time()
        join_time = end - start

        X, y = prepare_data_for_ml(dataframe=dataset_df, target_column=self.dataset.target_column)

        for model_name, training_fun in TRAINING_FUNCTIONS.items():
            if self.do_feature_selection:
                # ID3 not supported for feature selection
                if model_name == ID3:
                    continue
            print(f"==== Model Name: {model_name} ====")

            entry = Result(self.approach, self.dataset.base_table_id, self.dataset.base_table_label, model_name)
            accuracy, max_depth, feature_importances, train_time, sfs_time = hp_tune_join_all(
                X, y, training_fun, self.do_feature_selection
            )
            entry.set_depth(max_depth).set_accuracy(accuracy).set_feature_importance(
                feature_importances).set_train_time(train_time).set_join_time(join_time)
            if sfs_time:
                entry.set_feature_selection_time(sfs_time)
            self.results.append(entry)

        print(f"======== Finished dataset ========")