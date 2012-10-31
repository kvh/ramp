import sys
sys.path.append('../..')
from ramp import context, store
from ramp.features import base
from ramp.features.base import *
import unittest
from pandas import DataFrame, Series, Index
import pandas
import tempfile

from sklearn import linear_model
import numpy as np
import os, sys

from pandas.util.testing import assert_almost_equal


def strip_hash(s):
    return s[:-11]

def make_data(n):
        data = pandas.DataFrame(
                   np.random.randn(n, 3),
                   columns=['a','b','c'],
                   index=range(10, n+10))
        data['const'] = np.zeros(n)
        data['ints'] = range(n)
        data['y'] = data['a'] ** 2
        return data


class TestBasicFeature(unittest.TestCase):
    def setUp(self):
        self.data = make_data(10)

    def test_basefeature(self):
        f = BaseFeature('col1')
        self.assertEqual(f.feature, 'col1')
        self.assertEqual(str(f), 'col1')
        self.assertEqual(repr(f), "'col1'")
        self.assertEqual(f.unique_name, 'col1')

    def test_constantfeature_int(self):
        f = ConstantFeature(1)
        self.assertEqual(f.feature, 1)
        self.assertEqual(str(f), '1')
        self.assertEqual(repr(f), '1')
        self.assertEqual(f.unique_name, '1')

    def test_constantfeature_float(self):
        f = ConstantFeature(math.e)
        e_str = '2.71828182846'
        self.assertEqual(f.feature, math.e)
        self.assertEqual(str(f), e_str)
        self.assertEqual(repr(f), '2.718281828459045')
        self.assertEqual(f.unique_name, e_str)

    def test_combofeature(self):
        f = ComboFeature(['col1', 'col2'])
        for sf in f.features:
            self.assertIsInstance(sf, BaseFeature)
        self.assertEqual(str(f), 'Combo(col1, col2)')
        self.assertEqual(f.unique_name, 'Combo(col1, col2) [201b1e5d]')
        self.assertEqual(repr(f), "ComboFeature(_name='Combo',"
        "features=['col1', 'col2'])")

    def test_feature(self):
        f = Feature('col1')
        self.assertIsInstance(f.feature, BaseFeature)
        self.assertEqual(str(f), 'col1')
        self.assertEqual(f.unique_name, 'col1 [4e89804a]')
        self.assertEqual(repr(f), "Feature(_name='',"
        "feature='col1',features=['col1'])")

    def test_create_cache(self):
        f = base.Normalize(base.F(10) + base.F('a'))
        ctx = context.DataContext(store.MemoryStore('test', verbose=True), self.data)
        r = f.create(ctx)
        r = r[r.columns[0]]
        self.assertAlmostEqual(r.mean(), 0)
        self.assertAlmostEqual(r.std(), 1)

        # now add some new data
        idx = len(self.data) + 1000
        ctx.data = ctx.data.append(DataFrame([100, 200], columns=['a'], index=Index([idx, idx+1])))
        r = f.create(ctx)
        r = r[r.columns[0]]
        self.assertAlmostEqual(r[idx], (100 - self.data['a'].mean()) / self.data['a'].std())

        # drop all the other data ... should still use old prep data
        ctx.data = ctx.data.ix[[idx, idx+1]]
        r = f.create(ctx)
        r = r[r.columns[0]]
        self.assertAlmostEqual(r[idx], (100 - self.data['a'].mean()) / self.data['a'].std())



if __name__ == '__main__':
    unittest.main()

