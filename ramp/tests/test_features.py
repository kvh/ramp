import sys
sys.path.append('../..')
from ramp import context, store
from ramp.configuration import Configuration
from ramp.features import base
from ramp.features.base import *
from ramp.features.trained import *
import unittest
from pandas import DataFrame, Series, Index
import pandas
import tempfile

from sklearn import linear_model
from sklearn import decomposition
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
        self.ctx = context.DataContext(store.MemoryStore('test', verbose=True), self.data)

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
        ctx = self.ctx
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


class DummyEstimator(object):
    def __init__(self):
        pass

    def fit(self, x, y):
        self.fitx = x
        self.fity = y

    def predict(self, x):
        self.predictx = x
        p = np.zeros(len(x))
        return p


class DummyCVEstimator(object):
    def __init__(self):
        self.fitx = []
        self.fity = []
        self.predictx = []

    def fit(self, x, y):
        self.fitx.append(x)
        self.fity.append(y)

    def predict(self, x):
        self.predictx.append(x)
        p = np.zeros(len(x))
        return p


class TestTrainedFeature(unittest.TestCase):
    def setUp(self):
        self.data = make_data(10)
        self.ctx = context.DataContext(store.MemoryStore('test', verbose=True), self.data)

    def test_predictions(self):
        idx = 10
        est = DummyEstimator()
        f = Predictions(
                Configuration(target='y', features=[F('a')], model=est))
        r = f.create(self.ctx)
        r = r[r.columns[0]]
        assert_almost_equal(est.fitx.transpose()[0], self.data['a'].values)
        assert_almost_equal(est.predictx.transpose()[0], self.data['a'].values)

    def test_predictions_held_out(self):
        idx = 10
        est = DummyEstimator()
        f = Predictions(
                Configuration(target='y', features=[F('a')], model=est))
        self.ctx.train_index = self.ctx.train_index[:5]
        r = f.create(self.ctx)
        r = r[r.columns[0]]
        assert_almost_equal(est.fitx.transpose()[0], self.data['a'].values[:5])
        assert_almost_equal(est.predictx.transpose()[0], self.data['a'].values)

    def test_predictions_cv(self):
        idx = 10
        est = DummyCVEstimator()

        # make 2 folds
        folds = [(self.ctx.train_index[:4], self.ctx.train_index[4:8]),
                (self.ctx.train_index[4:8], self.ctx.train_index[:4])]

        f = Predictions(
                Configuration(target='y', features=[F('a')], model=est), cv_folds=folds)
        self.ctx.train_index = self.ctx.train_index[:8]
        r = f.create(self.ctx)
        r = r[r.columns[0]]

        # fit three times, one for each fold, one for held out data
        self.assertEqual(len(est.fitx), 3)
        assert_almost_equal(est.fitx[0].transpose()[0], self.data['a'].values[:4])
        assert_almost_equal(est.fitx[1].transpose()[0], self.data['a'].values[4:8])
        assert_almost_equal(est.fitx[2].transpose()[0], self.data['a'].values[:8])

        assert_almost_equal(est.predictx[0].transpose()[0], self.data['a'].values[4:8])
        assert_almost_equal(est.predictx[1].transpose()[0], self.data['a'].values[:4])
        assert_almost_equal(est.predictx[2].transpose()[0], self.data['a'].values[8:])


class TestGroupFeatures(unittest.TestCase):
    def setUp(self):
        self.data = make_data(10)
        self.data['groups'] = self.data['ints'].apply(lambda x: x > 5)
        self.ctx = context.DataContext(store.MemoryStore(verbose=True), self.data)

    def test_group_agg_col(self):
        f = GroupAggregate(['a', 'groups'], function=np.mean, data_column='a',
                groupby_column='groups')
        f.context = self.ctx

        # test prep data
        prep = f.get_prep_data(self.data)
        self.assertEqual(len(prep), 2)
        self.assertAlmostEqual(prep['global'], self.data['a'].mean())
        g1_mean = self.data['a'][self.data['groups']].mean()
        g2_mean = self.data['a'][-self.data['groups']].mean()
        self.assertAlmostEqual(prep['groups'].get(True), g1_mean)
        self.assertAlmostEqual(prep['groups'].get(False), g2_mean)

        # test feature creation
        data = get_single_column(f.create(self.ctx))
        expected = Series(index=data.index)
        expected[self.data['groups']] = g1_mean
        expected[-self.data['groups']] = g2_mean
        assert_almost_equal(data, expected)


class TestDimReduction(unittest.TestCase):
    def setUp(self):
        self.data = make_data(10)
        self.ctx = context.DataContext(data=self.data)

    def test_pca_dimred(self):
        decomposer = decomposition.PCA(n_components=2)
        f = combo.DimensionReduction(['a', 'b', 'c'],
                decomposer=decomposer)
        data = f.create(self.ctx)
        self.assertEqual(data.shape, (len(self.data), 2))
        decomposer = decomposition.PCA(n_components=2)
        expected = decomposer.fit_transform(self.data[['a', 'b', 'c']])
        assert_almost_equal(expected, data.values)



if __name__ == '__main__':
    unittest.main()

