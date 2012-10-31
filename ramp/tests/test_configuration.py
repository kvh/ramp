import sys
sys.path.append('../..')
from ramp.configuration import *
from ramp.features import *
from ramp.models import *
from ramp.metrics import *
import unittest
import pandas
from sklearn import linear_model
import numpy as np
import os, sys, random

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


if __name__ == '__main__':
    unittest.main()
