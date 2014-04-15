import tempfile
import sys
sys.path.append('../..')
from ramp.estimators import sk
from ramp import metrics
from ramp import *
import unittest
from pandas import *
from ramp import shortcuts

from ramp.estimators.base import Probabilities
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


if __name__ == '__main__':
    unittest.main()


