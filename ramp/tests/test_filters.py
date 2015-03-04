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
from ramp.modeling import generate_train
from ramp.tests.test_features import make_data
from ramp.transformations import inject_feature


class TestFilters(unittest.TestCase):
    def setUp(self):
        self.data = make_data(10)

    def test_discard_incomplete(self):
        model_def = ModelDefinition(features=[F('a'), Map('b', np.abs)],
                                    target='y',
                                    discard_incomplete=False)
        x, y, ff, ft = generate_train(model_def, self.data)
        self.assertEqual(len(x), len(self.data))
        # create incomplete cases
        self.data['a'][10] = None
        self.data['b'][11] = None
        self.data['b'][12] = None
        model_def = ModelDefinition(features=[F('a'), Map('b', np.abs)],
                                    target='y',
                                    discard_incomplete=True)
        x, y, ff, ft = generate_train(model_def, self.data)
        self.assertEqual(len(x), len(self.data) - 3)


if __name__ == '__main__':
    unittest.main()


