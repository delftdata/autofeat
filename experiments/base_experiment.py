from typing import List

from data_preparation.dataset_base import Dataset
from experiments.result_object import Result


class BaseExperiment:
    def __init__(self, data: Dataset, approach: str = '',  do_feature_selection=False, learning_curve_depth_values=None):
        self.dataset = data
        self.dataset.set_features()
        self.results: List[Result] = []
        self.learning_curve_depth_values = learning_curve_depth_values
        self.do_feature_selection = do_feature_selection
        self.approach = approach

    def compute_results(self):
        return

