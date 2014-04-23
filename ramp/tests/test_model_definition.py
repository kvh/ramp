import os
import pickle
import sys
sys.path.append('../..')
import unittest

import numpy as np
import pandas as pd
from pandas import DataFrame, Series, Index
from pandas.util.testing import assert_almost_equal
from sklearn import linear_model

from ramp.features.base import F, Map
from ramp.model_definition import ModelDefinition, model_definition_factory


class ModelDefinitionTest(unittest.TestCase):

    def test_model_def_factory(self):
        base = ModelDefinition(
                features=['a'],
                estimator='model',
                target='y'
                )
        factory = model_definition_factory(base,
            features=[
                ['a','b'],
                ['a','b','c'],
                ['a','b','c','y'],
                ],
            estimator=[
                'model2',
                'model3',
                ]
            )
        mds = list(factory)
        self.assertEqual(len(mds), 6)

    def test_model_def_pickle(self):
        c = ModelDefinition(
                features=['a', F('a'), Map('a', len)],
                estimator=linear_model.LogisticRegression()
                )
        s = pickle.dumps(c)
        c2 = pickle.loads(s)
        self.assertEqual(repr(c), repr(c2))

        # lambdas are not picklable, should fail
        c = ModelDefinition(
                features=['a', F('a'), Map('a', lambda x: len(x))],
                )
        self.assertRaises(pickle.PicklingError, pickle.dumps, c)


if __name__ == '__main__':
    unittest.main()
