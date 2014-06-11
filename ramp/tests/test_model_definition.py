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

from ramp.builders import build_featureset_safe
from ramp import estimators
from ramp.features.base import F, Map
from ramp.model_definition import ModelDefinition, model_definition_factory
from ramp.modeling import generate_train
from ramp.tests.test_features import make_data


class ModelDefinitionTest(unittest.TestCase):
    def setUp(self):
        self.data = make_data(10)

    def test_model_def_factory(self):
        base = ModelDefinition(
                features=['a'],
                estimator=estimators.Estimator('dummy'),
                target='y'
                )
        factory = model_definition_factory(base,
            features=[
                ['a','b'],
                ['a','b','c'],
                ['a','b','c','y'],
                ],
            estimator=[
                estimators.Estimator('dummy'),
                estimators.Estimator('dummy2'),
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
                estimator=linear_model.LogisticRegression()
                )
        self.assertRaises(pickle.PicklingError, pickle.dumps, c)

    def test_discard_incomplete(self):
        model_def = ModelDefinition(features=[F('a'), Map('b', np.abs)],
                                    target='y',
                                    discard_incomplete=False)
        self.assertEqual(model_def.filters, [])
        x, y, ff, ft = generate_train(model_def, self.data)
        self.assertEqual(len(x), len(self.data))

        # create incomplete cases
        self.data['a'][10] = None
        self.data['b'][11] = None
        self.data['b'][12] = None
        model_def = ModelDefinition(features=[F('a'), Map('b', np.abs)],
                                    target='y',
                                    discard_incomplete=True)
        self.assertEqual(len(model_def.filters), 1)
        x, y, ff, ft = generate_train(model_def, self.data)
        self.assertEqual(len(x), len(self.data) - 3)

    def test_categorical_indicators(self):
        self.data['categorical'] = map(str, range(10))
        model_def = ModelDefinition(features=[Map('categorical', list), F('a'), Map('b', np.abs)],
                                    target='y',
                                    categorical_indicators=False)
        x, ff = build_featureset_safe(model_def.features, self.data)
        self.assertEqual(len(x.columns), len(model_def.features))

        self.data['categorical'] = map(str, range(10))
        model_def = ModelDefinition(features=[Map('categorical', np.abs), F('a'), Map('b', np.abs)],
                                    target='y',
                                    categorical_indicators=True)
        x, ff = build_featureset_safe(model_def.features, self.data)
        self.assertEqual(len(x.columns), len(model_def.features) + 9)



if __name__ == '__main__':
    unittest.main()
