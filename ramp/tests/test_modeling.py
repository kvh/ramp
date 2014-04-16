import os
import sys
sys.path.append('../..')
import unittest

import numpy as np
import pandas as pd
from pandas import DataFrame, Series, Index
from pandas.util.testing import assert_almost_equal

from ramp.estimators.base import Probabilities
from ramp.features.base import F, Map
from ramp.features.trained import Predictions
from ramp.model_definition import ModelDefinition
from ramp.modeling import fit_model, predict_model
from ramp.tests.test_features import make_data


class DummyEstimator(object):
    def __init__(self):
        pass

    def fit(self, x, y):
        self.fitx = x
        self.fity = y

    def predict(self, x):
        self.predictx = x
        p = np.zeros(len(x))
        return p


class DummyCVEstimator(object):
    def __init__(self):
        self.fitx = []
        self.fity = []
        self.predictx = []

    def fit(self, x, y):
        self.fitx.append(x)
        self.fity.append(y)

    def predict(self, x):
        self.predictx.append(x)
        p = np.zeros(len(x))
        return p


class DummyProbEstimator(object):
    def __init__(self, n_clses):
        self.n_clses = n_clses

    def fit(self, x, y):
        pass

    def predict_proba(self, x):
        return np.zeros((len(x), self.n_clses))


class TestBasicModeling(unittest.TestCase):
    def setUp(self):
        self.data = make_data(10)

    def make_model_def_basic(self):
        features = [F(10), F('a')]
        target = F('b')
        estimator = DummyEstimator()

        model_def = ModelDefinition(features=features,
                                    estimator=estimator,
                                    target=target)
        return model_def

    def test_fit_model(self):
        model_def = self.make_model_def_basic()
        x, y, fitted_model = fit_model(model_def, self.data)
        fe = fitted_model.fitted_estimator
        self.assertEqual(fe.fitx.shape, x.shape)
        self.assertEqual(fe.fity.shape, y.shape)

    def test_predict_model(self):
        model_def = self.make_model_def_basic()
        x, y, fitted_model = fit_model(model_def, self.data)
        x, y_true, y_preds = predict_model(model_def, self.data[:3], fitted_model)
        self.assertEqual(len(x), 3)
        self.assertEqual(len(y_true), 3)
        self.assertEqual(len(y_preds), 3)


class TestNestedModeling(unittest.TestCase):
    def setUp(self):
        self.data = make_data(10)

    def test_predictions_nest(self):
        inner_estimator = DummyEstimator()
        inner_model = ModelDefinition(features=[F('a')],
                                      estimator=inner_estimator,
                                      target=F('b'))
        features = [F('c'), Predictions(inner_model)]
        target = F('b')
        estimator = DummyEstimator()

        model_def = ModelDefinition(features=features,
                                    estimator=estimator,
                                    target=target)

        x, y, fitted_model = fit_model(model_def, self.data, train_index=self.data.index[:5])
        self.assertEqual(fitted_model.fitted_features[1].trained_data.fitted_estimator.fitx.shape, (5, 1))
        self.assertEqual(x.shape, (len(self.data), 2))

        x, y_true, y_preds = predict_model(model_def, self.data[:3], fitted_model)
        assert_almost_equal(x[x.columns[1]].values, np.zeros(3))
        

if __name__ == '__main__':
    unittest.main()


