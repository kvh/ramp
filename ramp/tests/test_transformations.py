import os
import sys
sys.path.append('../..')
import unittest

import numpy as np
import pandas as pd
from pandas import DataFrame, Series, Index
from pandas.util.testing import assert_almost_equal

from ramp.features.base import F, Map
from ramp.model_definition import ModelDefinition
from ramp.tests.test_features import make_data
from ramp.transformations import inject_feature


class TestTransformations(unittest.TestCase):
    def setUp(self):
        self.data = make_data(10)

    def test_inject_feature_simple(self):
        f = Map(Map(F('a'), np.abs), np.abs)
        self.assertEqual(str(f), 'absolute(absolute(a))')
        new_f = inject_feature(f, Map, function=lambda x: x + 1000)
        self.assertEqual(str(new_f), 'absolute(absolute(<lambda>(a)))')

    def test_inject_feature_combo(self):
        f = Map(F('a') + F('b'), np.abs)
        self.assertEqual(str(f), 'absolute(Add(a, b))')
        new_f = inject_feature(f, Map, function=lambda x: x + 1000)
        # Should inject in
        self.assertEqual(str(new_f), 'absolute(Add(<lambda>(a), <lambda>(b)))')

if __name__ == '__main__':
    unittest.main()


