import sys
sys.path.append('../..')
from ramp.configuration import *
from ramp.features import *
from ramp.dataset import *
from ramp.models import *
from ramp.metrics import *
from ramp import core
import unittest
import pandas
from sklearn import linear_model
import numpy as np
import os, sys, random

from pandas.util.testing import assert_almost_equal

def nothing(x): return x
def something(x): return 2 * x

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
        # expected = [[f.unique_name for f in x] for x in [
        #         [[F('a'),F('b')]] * 3 + [[F('a'),F('b'),F('c')]] * 3]
        # actual = [[f.unique_name for f in cnf.features] for cnf in cnfs]
        # for f, m in zip(actual, [cnf.model for cnf in cnfs]):
        #     print f, m
        # self.assertEqual(expected, actual)



if __name__ == '__main__':
    unittest.main()
