
class Result:
    TFD_PATH = "tfd-path"
    TFD = 'tfd'
    ARDA = 'arda'
    JOIN_ALL = "join-all"
    JOIN_ALL_FS = "join-all-fs"
    BASE = "base-table"

    def __init__(self, approach, data_path, algorithm, join_time=None):
        self.approach = approach
        self.data_path = data_path
        self.algorithm = algorithm
        self.join_time = join_time
        self.total_time = join_time if join_time else 0
        self.feature_selection_time = None
        self.depth = None
        self.accuracy = None
        self.train_time = None
        self.feature_importance = None

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
