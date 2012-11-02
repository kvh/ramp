import sys
sys.path.append('../..')
from ramp.configuration import *
from ramp.features import *
from ramp.features.base import *
from ramp.models import *
from ramp.metrics import *
import unittest
import pandas
from sklearn import linear_model
import numpy as np
import os, sys, random, pickle

from pandas.util.testing import assert_almost_equal


class ConfigurationTest(unittest.TestCase):

    def test_config_factory(self):
        base = Configuration(
                features=['a'],
                model='model',
                target='y'
                )
        fact = ConfigFactory(base,
            features=[
                ['a','b'],
                ['a','b','c'],
                ],
            model=[
                'model2',
                'model3',
                ]
            )
        cnfs = [cnf for cnf in fact]
        self.assertEqual(len(cnfs), 4)

    def test_configuration_pickle(self):
        c = Configuration(
                features=['a', F('a'), Map('a', len)],
                model=linear_model.LogisticRegression()
                )
        s = pickle.dumps(c)
        c2 = pickle.loads(s)
        self.assertEqual(repr(c), repr(c2))

        # lambdas are not picklable, should fail
        c = Configuration(
                features=['a', F('a'), Map('a', lambda x: len(x))],
                )
        self.assertRaises(pickle.PicklingError, pickle.dumps, c)

if __name__ == '__main__':
    unittest.main()
