import sys
sys.path.append('../..')
import unittest

import numpy as np
import pandas as pd
from pandas import DataFrame, Series, Index
from pandas.util.testing import assert_almost_equal

from ramp.builders import *
from ramp.estimators.base import Probabilities
from ramp.features.base import F, Map
from ramp.features.trained import Predictions
from ramp.model_definition import ModelDefinition
from ramp.selectors import BinaryFeatureSelector
from ramp.tests.test_features import make_data


class TestSelectors(unittest.TestCase):
    def setUp(self):
        self.data = make_data(100)

    def test_binary_selectors(self):
        d = self.data
        d['target'] = [0] * 50 + [1] * 50
        d['good_feature'] = [0] * 35 + [1] * 65
        d['best_feature'] = d['target']
        features = map(F, ['a', 'b', 'good_feature', 'best_feature'])
        selector = BinaryFeatureSelector()
        x, ffs = build_featureset_safe(features, self.data)
        y, ff = build_target_safe(F('target'), self.data)
        cols = selector.select(x, y, 2)
        feature_rank = [F('best_feature'), F('good_feature')]
        self.assertEqual(cols, [f.unique_name for f in feature_rank])

    def test_binary_selectors_multiclass(self):
        d = self.data
        d['target'] = [0] * 50 + [1] * 25 + [2] * 25
        d['good_feature'] = [0] * 35 + [1] * 65
        d['best_feature'] = d['target']
        features = map(F, ['a', 'b', 'good_feature', 'best_feature'])
        selector = BinaryFeatureSelector()
        x, ffs = build_featureset_safe(features, self.data)
        y, ff = build_target_safe(F('target'), self.data)
        cols = selector.select(x, y, 2)
        feature_rank = [F('best_feature'), F('good_feature')]
        self.assertEqual(cols, [f.unique_name for f in feature_rank])


if __name__ == '__main__':
    unittest.main()
