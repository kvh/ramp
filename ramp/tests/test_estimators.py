import sys
sys.path.append('../..')
import unittest

import numpy as np
import pandas as pd
from sklearn import linear_model

from ramp.estimators import (Probabilities,
                             BinaryProbabilities,
                             wrap_sklearn_like_estimator)
from ramp import shortcuts
from ramp.tests.test_features import make_data


class DummyProbEstimator(object):
    def __init__(self, n_clses):
        self.n_clses = n_clses
        self._coefs = "coefs"

    def fit(self, x, y):
        pass

    def predict_proba(self, x):
        return np.zeros((len(x), self.n_clses))


class TestEstimators(unittest.TestCase):
    def setUp(self):
        self.data = make_data(10)

    def test_probabilities(self):
        inner_est = DummyProbEstimator(3)
        est = wrap_sklearn_like_estimator(inner_est)

        # test attr wrap
        self.assertEqual(est._coefs, inner_est._coefs)
        self.assertRaises(AttributeError, getattr, est, 'nope_not_attr')

        preds = est.predict(self.data.values)        
        self.assertEqual(preds.shape, (10, 3))

    def test_binary_probabilities(self):
        inner_est = DummyProbEstimator(2)
        est = wrap_sklearn_like_estimator(inner_est)

        # test attr wrap
        self.assertEqual(est._coefs, inner_est._coefs)

        preds = est.predict(self.data.values)        
        self.assertEqual(preds.shape, (10, ))

    def test_sklearn_probabilities(self):
        # test multi-class
        self.data['target'] = [0] * 5 + [1] * 3 + [2] * 2
        inner_est = linear_model.LogisticRegression()
        est = wrap_sklearn_like_estimator(inner_est)
        x = self.data[['a', 'b']]
        est.fit(x, self.data.target)
        preds = est.predict(x)
        self.assertEqual(preds.shape, (10, 3))

        # test binary, single output
        self.data['target'] = [0] * 5 + [1] * 5
        est = BinaryProbabilities(inner_est)
        x = self.data[['a', 'b']]
        est.fit(x, self.data.target)
        preds = est.predict(x)
        self.assertEqual(preds.shape, (10, ))
        

if __name__ == '__main__':
    unittest.main()

