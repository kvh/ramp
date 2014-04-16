import sys
sys.path.append('../..')
import unittest

import pandas as pd

from ramp.estimators import Probabilities, BinaryProbabilities
from ramp import shortcuts
from ramp.tests.test_features import make_data


class DummyProbEstimator(object):
    def __init__(self, n_clses):
        self.n_clses = n_clses

    def fit(self, x, y):
        pass

    def predict_proba(self, x):
        return np.zeros((len(x), self.n_clses))


class TestSKEstimators(unittest.TestCase):
    def setUp(self):
        self.data = make_data(10)

    def test_probabilities(self):
        est = Probabilities(DummyProbEstimator(2))
        result = shortcuts.predict(
                store=store.MemoryStore(),
                data=self.data, model=est,
                predict_index=self.data.index,
                target='y', metrics=[metrics.AUC()], features=['a'])
        self.assertEqual(preds.shape, (10, 2))

    def test_binary_probabilities(self):
        est = BinaryProbabilities(DummyProbEstimator(2))
        result = shortcuts.predict(
                store=store.MemoryStore(),
                data=self.data, model=est,
                predict_index=self.data.index,
                target='y', metrics=[metrics.AUC()], features=['a'])
        self.assertEqual(len(result['predictions']), 10)
        t = np.zeros(10)
        t[0] = 1
        metrics.AUC().score(t, result['predictions'])


if __name__ == '__main__':
    unittest.main()

