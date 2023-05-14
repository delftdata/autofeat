import unittest
import feature_discovery.arda.arda as arda
import numpy as np
import logging
import sys


class MockRegressor():
    count = 0
    def fit(self, X, y):
        res = np.zeros(X.shape[1])
        res[MockRegressor.count % X.shape[1]] = 1
        MockRegressor.count = MockRegressor.count + 1
        self.feature_importances_ = res


class MockEstimator():
    count = 0
    def fit(self, X, y):
        MockEstimator.count += 1

    def score(self, X, y):
        return 1.0 - 0.1 * MockEstimator.count
    

class TestARDA(unittest.TestCase):

    def test_gen_features_size_zero_and_below(self):
        A = np.ones((2 , 4))
        features = arda.gen_features(A, 0)
        features_neg = arda.gen_features(A, -1)
        self.assertEqual(features.shape, (0, ))
        self.assertEqual(features.shape, (0, ))


    def test_gen_features_size_above_zero(self):
        A = np.ones((2 , 4))
        count = 3
        features = arda.gen_features(A, count)
        self.assertEqual(features.shape, (A.shape[0], count * A.shape[1]))


    def test_bin_count(self):
        importances = np.array([2, 4, 1, 3])
        mask = np.array([False, False, True, True], dtype=bool)
        bin_size = 2
        counts = arda._bin_count_ranking(importances, mask, bin_size)
        # second element has highest count in the importances, so that one should be incremented
        self.assertEqual(list(counts), [0, 1])
        self.assertEqual(type(counts), np.ndarray)  # Test whether an array is returned, since we use addition for incrementing


    def test_bin_count_zero(self):
        importances = np.array([1, 2, 3, 4])
        mask = np.array([False, False, True, True], dtype=bool)
        bin_size = 2
        counts = arda._bin_count_ranking(importances, mask, bin_size)
        self.assertEqual(list(counts), [0, 0])


    def test_select_features_k(self):
        MockRegressor.count = 0
        A = np.ones((2 , 4))
        result = arda.select_features(A, A[:, 0], tau=0, eta=1, k=2, regressor=MockRegressor)
        self.assertEqual(list(result), [0, 1])  # Since we add one to every feature sequentially in the mock regressor, for k=2 the first two should come out


    def test_select_features_k_zero(self):
        MockRegressor.count = 0
        A = np.ones((2 , 4))
        result = arda.select_features(A, A[:, 0], tau=0, eta=1, k=0, regressor=MockRegressor)
        self.assertEqual(list(result), [])  # Should be empty for k=0


    def test_select_features_tau_high(self):
        MockRegressor.count = 0
        A = np.ones((2 , 4))
        result = arda.select_features(A, A[:, 0], tau=0.5, eta=1, k=2, regressor=MockRegressor)
        self.assertEqual(list(result), [])  # Should be empty since 1/k = 0.5, counts are normalized for runs and we count one per feature here


    def test_select_features_tau_low(self):
        MockRegressor.count = 0
        A = np.ones((2 , 4))
        result = arda.select_features(A, A[:, 0], tau=0.49, eta=1, k=2, regressor=MockRegressor)
        self.assertEqual(list(result), [0, 1])  # Should be first two again, since 1/k = 0.5


    def test_select_features_all(self):
        MockRegressor.count = 0
        A = np.ones((2 , 4))
        result = arda.select_features(A, A[:, 0], tau=0, eta=1, k=A.shape[1], regressor=MockRegressor)
        self.assertEqual(list(result), [0, 1, 2, 3])  # Should be all features in A (which are 4)


    def test_wrapper_algo(self):
        MockRegressor.count = 0
        MockEstimator.count = 0

        A = np.ones((4 , 8))
        indices = arda.wrapper_algo(A, A[:, 0], [0, 0, 0], eta=1, k=2, estimator=MockEstimator, regressor=MockRegressor)
        self.assertEqual(MockEstimator.count, 2)  # Should have ran two times, since accuracy decreases after first run due to mock estimator


if __name__ == '__main__':
    logging.basicConfig( stream=sys.stderr )
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)