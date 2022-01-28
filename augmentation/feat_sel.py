from ITMO_FS import su_measure, gini_index, information_gain, spearman_corr, reliefF_measure


class FeatSel:
    SU = 'symmetrical uncertainty'
    GINI = 'gini-index'
    GAIN = 'information-gain'
    CORR = 'spearman-correlation'
    RELIEF = 'reliefF'

    def feature_selection(self, selection_method, X, y):
        feat_sel = self._get_feat_sel(selection_method)
        return feat_sel(X, y)

    def _get_feat_sel(self, selection_method):
        if selection_method == self.SU:
            return self.su
        elif selection_method == self.GINI:
            return self.gini
        elif selection_method == self.GAIN:
            return self.gain
        elif selection_method == self.CORR:
            return self.correlation
        elif selection_method == self.RELIEF:
            return self.relief
        else:
            raise ValueError(selection_method)

    def su(self, X, y):
        return su_measure(X, y)

    def gini(self, X, y):
        return gini_index(X, y)

    def gain(self, X, y):
        return information_gain(X, y)

    def correlation(self, X, y):
        return spearman_corr(X, y)

    def relief(self, X, y):
        return reliefF_measure(X, y)
