
class Result:
    TFD_PATH = "TFD_PATH"
    TFD = 'TFD'
    ARDA = 'ARDA'
    JOIN_ALL = "TFD_JOIN_ALL"
    JOIN_ALL_FS = "TFD_JOIN_ALL_FS"
    BASE = "BASE"

    def __init__(self, approach, data_path, dataset_label, algorithm, join_time=None):
        self.approach = approach
        self.data_path = data_path
        self.dataset = dataset_label
        self.algorithm = algorithm
        self.join_time = join_time
        self.total_time = join_time if join_time else 0
        self.feature_selection_time = None
        self.depth = None
        self.accuracy = None
        self.train_time = None
        self.feature_importance = None
        self.cutoff_threshold = None
        self.redundancy_threshold = None

    def set_cutoff_threshold(self, cutoff_threshold):
        self.cutoff_threshold = cutoff_threshold
        return self

    def set_redundancy_threshold(self, redundancy_threshold):
        self.redundancy_threshold = redundancy_threshold
        return self

    def set_depth(self, depth):
        self.depth = depth
        return self

    def set_accuracy(self, accuracy):
        self.accuracy = accuracy
        return self

    def set_train_time(self, train_time):
        self.train_time = train_time
        self.total_time += train_time
        return self

    def set_feature_importance(self, feature_importance):
        self.feature_importance = feature_importance
        return self

    def set_feature_selection_time(self, feature_selection_time):
        self.feature_selection_time = feature_selection_time
        self.total_time += feature_selection_time
        return self
