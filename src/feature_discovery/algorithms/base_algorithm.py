

class BaseAlgorithm:
    LABEL = "algorithm_name"

    def __init__(self):
        self.algorithm = None

    def train(self, X, y, do_feature_selection: bool = False):
        return

