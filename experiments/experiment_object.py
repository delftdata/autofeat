
class Experiment:
    def __init__(self, data_path, data_label, model, accuracy, approach):
        self.data_path = data_path
        self.data_label = data_label
        self.model = model
        self.accuracy = accuracy
        self.approach = approach
        self.depth = None
        self.rank = None
        self.cutoff_th = None
        self.redundancy_th = None
        self.features = None

    def set_features(self, features):
        self.features = features
        return self

    def set_depth(self, depth):
        self.depth = depth
        return self

    def set_rank(self, rank):
        self.rank = rank
        return self

    def set_cutoff_th(self, cutoff_th):
        self.cutoff_th = cutoff_th
        return self

    def set_redundancy_th(self, redundancy_th):
        self.redundancy_th = redundancy_th
        return self


