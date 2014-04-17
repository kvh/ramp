import os
import sys
sys.path.append('../..')
import unittest

import numpy as np
import pandas as pd
from pandas import DataFrame, Series, Index
from pandas.util.testing import assert_almost_equal
from sklearn import linear_model

from ramp.estimators.base import Probabilities
from ramp.features.base import F, Map
from ramp.features.trained import Predictions
from ramp.model_definition import ModelDefinition
from ramp.shortcuts import cross_validate, cv_factory
from ramp.tests.test_features import make_data


class TestShortcuts(unittest.TestCase):
    def setUp(self):
        self.data = make_data(10)

    def test_cross_validate(self):
        results, reports = cross_validate(self.data, folds=3, 
                                          features = [F(10), F('a')],
                                          target = F('b'),
                                          estimator = linear_model.LinearRegression())
        self.assertEqual(len(results), 3)

    def test_cross_validate_factory(self):
        results, reports = cv_factory(self.data, folds=3, 
                                          features=[[F(10), F('a')]],
                                          target=[F('b'), F('a')],
                                          estimator=[linear_model.LinearRegression()])
        self.assertEqual(len(results), 3)

if __name__ == '__main__':
    unittest.main()


