import sys
sys.path.append('../..')
from ramp import context, store
from ramp.features import base
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
