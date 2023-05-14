import time

from feature_discovery.algorithms import TRAINING_FUNCTIONS, ID3
from feature_discovery.data_preparation.dataset_base import Dataset
from feature_discovery.data_preparation.join_data import join_all
from feature_discovery.data_preparation.utils import prepare_data_for_ml
from feature_discovery.experiments.base_experiment import BaseExperiment
from feature_discovery.experiments.result_object import Result
from feature_discovery.experiments.utils import hp_tune_join_all


class JoinAllExperiment(BaseExperiment):
    def __init__(self, data: Dataset, learning_curve_depth_values=None, do_feature_selection=False):
        super().__init__(data, approach=Result.JOIN_ALL_FS if do_feature_selection else Result.JOIN_ALL,
                         do_feature_selection=do_feature_selection,
                         learning_curve_depth_values=learning_curve_depth_values)

    def compute_results(self):
        print(f'======== JOIN-ALL Dataset Pipeline - feature selection - {self.do_feature_selection} ========')
        start = time.time()
        dataset_df = join_all(self.dataset.base_table_id)
        end = time.time()
        join_time = end - start

        X, y = prepare_data_for_ml(dataframe=dataset_df, target_column=self.dataset.target_column)

        for algorithm in TRAINING_FUNCTIONS:
            if self.do_feature_selection:
                # ID3 not supported for feature selection
                if algorithm.LABEL == ID3.LABEL:
                    continue
            print(f"==== Model Name: {algorithm.LABEL} ====")

            accuracy, max_depth, feature_importances, train_time, sfs_time = hp_tune_join_all(
                X, y, algorithm().train, self.do_feature_selection
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
                feature_selection_time=sfs_time,
                join_time=join_time,
            )
            self.results.append(entry)

        print(f"======== Finished dataset ========")


class JoinAllFeatureSelectionExperiment(JoinAllExperiment):
    def __init__(self, dataset: Dataset, learning_curve_depth_values=None):
        super().__init__(dataset, do_feature_selection=True, learning_curve_depth_values=learning_curve_depth_values)
