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
from ramp import modeling
from ramp.modeling import (fit_model,
                           cross_validate,
                           build_and_package_model,
                           generate_train,
                           generate_test)
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

    def test_generate_train(self):
        model_def = self.make_model_def_basic()
        train_index = self.data.index[:5]
        x_train, y_train, ff, x = generate_train(model_def, self.data,
                                          train_index=train_index)
        assert_almost_equal(x_train.index, train_index)

    def test_generate_test(self):
        model_def = self.make_model_def_basic()
        x, y, fitted_model = fit_model(model_def, self.data)
        test_index = self.data.index[:5]
        x_test, y_test = generate_test(model_def, self.data.loc[test_index],
                                       fitted_model)
        assert_almost_equal(x_test.index, test_index)

    def test_fit_model(self):
        model_def = self.make_model_def_basic()
        x, y, fitted_model = fit_model(model_def, self.data)
        fe = fitted_model.fitted_estimator
        self.assertEqual(fe.fitx.shape, x.shape)
        self.assertEqual(fe.fity.shape, y.shape)

    def test_predict(self):
        model_def = self.make_model_def_basic()
        x, y, fitted_model = fit_model(model_def, self.data)
        x, y_true = modeling.generate_test(model_def, self.data[:3], fitted_model)
        y_preds = fitted_model.fitted_estimator.predict(x)
        self.assertEqual(len(x), 3)
        self.assertEqual(len(y_true), 3)
        self.assertEqual(len(y_preds), 3)
        y_preds2 = modeling.predict_with_model(fitted_model, self.data[:3])
        assert_almost_equal(y_preds, y_preds2.values)

    def test_cross_validate(self):
        model_def = self.make_model_def_basic()
        results = cross_validate(model_def, self.data, folds=3)
        self.assertEqual(len(results), 3)

    def test_build_and_package_model(self):
        model_def = self.make_model_def_basic()
        desc =  "State-of-the-Art Model"
        pkg = build_and_package_model(model_def, self.data, desc,
                                      train_index=self.data.index[:3])
        self.assertEqual(pkg.data_description, desc)
        self.assertTrue(pkg.fitted_model)

        # and evaluate
        pkg = build_and_package_model(model_def, self.data, desc, evaluate=True,
                                      train_index=self.data.index[:3])
        self.assertEqual(pkg.data_description, desc)
        self.assertTrue(pkg.fitted_model)

    def test_cross_validate_with_alternative_predictions(self):
        features = [F(10), F('a')]
        target = Map('b', np.log)
        estimator = DummyEstimator()

        model_def = ModelDefinition(features=features,
                                    estimator=estimator,
                                    target=target,
                                    evaluation_transformation=Map('__predictions', np.exp),
                                    evaluation_target=F('b'))
        results = cross_validate(model_def, self.data, folds=3)
        self.assertEqual(len(results), 3)
        # assert we've transformed predictions correctly
        yt = results[0].y_test
        assert_almost_equal(yt.values, self.data['b'].reindex(yt.index).values)
        assert_almost_equal(estimator.fity,
                            np.log(self.data['b'].reindex(results[-1].y_train.index)).values)


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

        train_index = self.data.index[:5]
        x, y, fitted_model = fit_model(model_def, self.data, train_index=train_index)
        self.assertEqual(fitted_model.fitted_features[1].trained_data.fitted_estimator.fitx.shape, (5, 1))
        self.assertEqual(x.shape, (len(train_index), 2))

        x, y_true = modeling.generate_test(model_def, self.data[:3], fitted_model)
        assert_almost_equal(x[x.columns[1]].values, np.zeros(3))



if __name__ == '__main__':
    unittest.main()


