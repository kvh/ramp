import os
import sys
sys.path.append('../..')
import unittest

import numpy as np
import pandas as pd
from pandas import DataFrame, Series, Index
from pandas.util.testing import assert_almost_equal
from sklearn import decomposition

from ramp.builders import *
from ramp.features import base, text
from ramp.features.base import *
from ramp.features.trained import *
from ramp.model_definition import ModelDefinition


def strip_hash(s):
    return s[:-11]

def make_data(n):
    data = pd.DataFrame(
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

    def test_basefeature_reprs(self):
        f = BaseFeature('col1')
        self.assertFalse(f.is_trained)
        self.assertFalse(f.is_prepped)
        self.assertEqual(f.feature, 'col1')
        self.assertEqual(str(f), 'col1')
        self.assertEqual(repr(f), "'col1'")
        self.assertEqual(f.unique_name, 'col1')

    def test_constantfeature_int_reprs(self):
        f = ConstantFeature(1)
        self.assertFalse(f.is_trained)
        self.assertFalse(f.is_prepped)
        self.assertEqual(f.feature, 1)
        self.assertEqual(str(f), '1')
        self.assertEqual(repr(f), '1')
        self.assertEqual(f.unique_name, '1')

    def test_constantfeature_float_reprs(self):
        f = ConstantFeature(math.e)
        self.assertEqual(f.feature, math.e)
        self.assertEqual(str(f), str(math.e))
        self.assertEqual(repr(f), repr(math.e))
        print f.unique_name
        self.assertEqual(f.unique_name, str(math.e))

    def test_combofeature_reprs(self):
        f = ComboFeature(['col1', 'col2'])
        for sf in f.features:
            self.assertIsInstance(sf, BaseFeature)
        self.assertEqual(str(f), 'Combo(col1, col2)')
        self.assertEqual(f.unique_name, 'Combo(col1, col2) [201b1e5d]')
        self.assertEqual(repr(f), "ComboFeature(_name='Combo',"
        "features=['col1', 'col2'])")

    def test_feature_reprs(self):
        f = Feature('col1')
        self.assertFalse(f.is_trained)
        self.assertFalse(f.is_prepped)
        self.assertIsInstance(f.feature, BaseFeature)
        self.assertEqual(str(f), 'col1')
        self.assertEqual(f.unique_name, 'col1 [4e89804a]')
        self.assertEqual(repr(f), "Feature(_name='',"
        "feature='col1',features=['col1'])")

    def test_basic_feature_chaining(self):
        a_mean = self.data.a.mean()
        f = base.Normalize(base.F(10) + base.F('a'))

        # test build
        res, fitted_feature = f.build(self.data)
        self.assertEqual(len(fitted_feature.prepped_data), 1)
        vals = fitted_feature.prepped_data.values()[0]
        self.assertAlmostEqual(vals[0], 10 + a_mean)
        self.assertAlmostEqual(vals[1], self.data.a.std())

        # test fitted feature
        self.assertEqual(len(fitted_feature.inner_fitted_features), 1)
        iff = fitted_feature.inner_fitted_features[0]
        self.assertEqual(len(iff.inner_fitted_features), 2)
        iiff = iff.inner_fitted_features[0]
        self.assertTrue(iiff.inner_fitted_features[0] is None)

        # test apply
        res = f.apply(self.data, fitted_feature)
        self.assertAlmostEqual(res[res.columns[0]].mean(), 0)
        self.assertAlmostEqual(res[res.columns[0]].std(), 1)

        # test callable functionality
        res = f(self.data, fitted_feature)
        self.assertAlmostEqual(res[res.columns[0]].mean(), 0)
        self.assertAlmostEqual(res[res.columns[0]].std(), 1)

        # test no side-effects
        self.assertAlmostEqual(a_mean, self.data.a.mean())

    def test_basic_feature_prep_index(self):
        a_mean = self.data.a.mean()
        f = base.Normalize(base.F(10) + base.F('a'))
        prep_data = self.data.iloc[range(len(self.data) / 2)]
        self.assertFalse(f.is_trained)
        self.assertTrue(f.is_prepped)

        # test build
        res, fitted_feature = f.build(self.data, prep_index=prep_data.index)
        self.assertEqual(len(fitted_feature.prepped_data), 1)
        vals = fitted_feature.prepped_data.values()[0]
        self.assertAlmostEqual(vals[0], 10 + prep_data.a.mean())
        self.assertAlmostEqual(vals[1], prep_data.a.std())

        # test apply
        res = f.apply(self.data, fitted_feature)
        expected = (self.data.a - vals[0] + 10) / vals[1]
        assert_almost_equal(res[res.columns[0]].values, expected.values)

    def test_feature_builders(self):
        # good features
        features = [base.F(10), base.F('a')]
        featureset, fitted_features = build_featureset_safe(features, self.data)
        self.assertEqual(featureset.shape, (len(self.data), 2))
        featureset = apply_featureset_safe(features, self.data, fitted_features)
        self.assertEqual(featureset.shape, (len(self.data), 2))

        # bad feature, drops data
        class BuggyFeature(base.F):
            def _apply(self, data, fitted_feature):
                return data.iloc[:len(data)/2]
        features = [base.F(10), base.F('a'), BuggyFeature('a')]
        with self.assertRaises(AssertionError):
            featureset, ffs = build_featureset_safe(features, self.data)

        # target
        featureset, fitted_feature = build_target_safe(base.F('a'), self.data)
        self.assertEqual(featureset.shape, (len(self.data), ))
        self.assertTrue(isinstance(featureset, Series))
        featureset = apply_target_safe(base.F('a'), self.data, fitted_feature)
        self.assertEqual(featureset.shape, (len(self.data), ))
        self.assertTrue(isinstance(featureset, Series))


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

    def make_model_def_basic(self):
        features = [F(10), F('a')]
        target = F('b')
        estimator = DummyEstimator()

        model_def = ModelDefinition(features=features,
                                    estimator=estimator,
                                    target=target)
        return model_def

    def test_predictions(self):
        model_def = self.make_model_def_basic()
        f = Predictions(model_def)
        self.assertTrue(f.is_trained)
        self.assertFalse(f.is_prepped)

        r, ff = f.build(self.data)
        r = r[r.columns[0]]
        assert_almost_equal(r.values, np.zeros(len(self.data)))
        fitted_model = ff.trained_data
        #TODO uggh fix this
        print fitted_model.fitted_estimator.fitx
        assert_almost_equal(fitted_model.fitted_estimator.fitx.transpose()[1], self.data['a'].values)
        assert_almost_equal(fitted_model.fitted_estimator.predictx.transpose()[1], self.data['a'].values)

    def test_predictions_held_out(self):
        model_def = self.make_model_def_basic()
        f = Predictions(model_def)
        r, ff = f.build(self.data, train_index=self.data.index[:5])
        r = r[r.columns[0]]
        assert_almost_equal(r.values, np.zeros(len(self.data)))
        fitted_model = ff.trained_data
        assert_almost_equal(fitted_model.fitted_estimator.fitx.transpose()[1], self.data['a'].values[:5])
        assert_almost_equal(fitted_model.fitted_estimator.predictx.transpose()[1], self.data['a'].values)

    def test_residuals(self):
        model_def = self.make_model_def_basic()
        f = Residuals(model_def)
        r, ff = f.build(self.data)
        r = r[r.columns[0]]
        assert_almost_equal(r.values, 0 - self.data['b'].values)
        fitted_model = ff.trained_data
        assert_almost_equal(fitted_model.fitted_estimator.fitx.transpose()[1], self.data['a'].values)
        assert_almost_equal(fitted_model.fitted_estimator.predictx.transpose()[1], self.data['a'].values)

    # def test_predictions_cv(self):
    #     idx = 10
    #     est = DummyCVEstimator()

    #     # make 2 folds
    #     folds = [(self.ctx.train_index[:4], self.ctx.train_index[4:8]),
    #             (self.ctx.train_index[4:8], self.ctx.train_index[:4])]

    #     f = Predictions(
    #             Configuration(target='y', features=[F('a')], model=est), cv_folds=folds)
    #     self.ctx.train_index = self.ctx.train_index[:8]
    #     r = f.create(self.ctx)
    #     r = r[r.columns[0]]

    #     # fit three times, one for each fold, one for held out data
    #     self.assertEqual(len(est.fitx), 3)
    #     assert_almost_equal(est.fitx[0].transpose()[0], self.data['a'].values[:4])
    #     assert_almost_equal(est.fitx[1].transpose()[0], self.data['a'].values[4:8])
    #     assert_almost_equal(est.fitx[2].transpose()[0], self.data['a'].values[:8])

    #     assert_almost_equal(est.predictx[0].transpose()[0], self.data['a'].values[4:8])
    #     assert_almost_equal(est.predictx[1].transpose()[0], self.data['a'].values[:4])
    #     assert_almost_equal(est.predictx[2].transpose()[0], self.data['a'].values[8:])

    def test_target_aggregation_by_factor(self):
        self.data['grp'] = [0] * 5 + [1] * (len(self.data) - 5)
        f = TargetAggregationByFactor(group_by='grp', func=np.mean, target='ints', min_sample=1)
        d, ff = f.build(self.data, train_index=self.data.index[:6])
        keys, vals = ff.trained_data
        self.assertEqual(len(keys), 2)
        self.assertAlmostEqual(vals[0], np.mean(range(5)))
        self.assertAlmostEqual(vals[1], np.mean(5))



def make_text_data(n):
    data = pd.DataFrame(
               columns=['a','b','c'],
               index=range(10, n+10))
    data['a'] = [' '.join([chr(random.randint(97,123))*5 for i in range(5)]) for _ in range(n)]
    data['b'] = [' '.join([chr(random.randint(97,123))*5 for i in range(2)]) for _ in range(n)]
    data['c'] = [' '.join([chr(random.randint(97,123))*2 for i in range(2)]) for _ in range(n)]
    return data


class TestTextFeatures(unittest.TestCase):
    def setUp(self):
        self.n = 1000
        self.data = make_text_data(self.n)

    def test_LSI_topics(self):
        f = text.LSI(text.Tokenizer('a'), num_topics=1)
        feature_data, ff = f.build(self.data)
        self.assertEqual(feature_data.shape, (self.n, 1))

    def test_ngramcounts(self):
        f = text.NgramCounts(text.Tokenizer('a'), mindocs=1)
        feature_data, ff = f.build(self.data)
        self.assertEqual(feature_data.shape, (self.n, 26))


# class TestGroupFeatures(unittest.TestCase):
#     def setUp(self):
#         self.data = make_data(10)
#         self.data['groups'] = self.data['ints'].apply(lambda x: x > 5)
#         self.ctx = context.DataContext(store.MemoryStore(verbose=True), self.data)

#     def test_group_agg_col(self):
#         f = GroupAggregate(['a', 'groups'], function=np.mean, data_column='a',
#                 groupby_column='groups')
#         f.context = self.ctx

#         # test prep data
#         prep = f.get_prep_data(self.data)
#         self.assertEqual(len(prep), 2)
#         self.assertAlmostEqual(prep['global'], self.data['a'].mean())
#         g1_mean = self.data['a'][self.data['groups']].mean()
#         g2_mean = self.data['a'][-self.data['groups']].mean()
#         self.assertAlmostEqual(prep['groups'].get(True), g1_mean)
#         self.assertAlmostEqual(prep['groups'].get(False), g2_mean)

#         # test feature creation
#         data = get_single_column(f.create(self.ctx))
#         expected = Series(index=data.index)
#         expected[self.data['groups']] = g1_mean
#         expected[-self.data['groups']] = g2_mean
#         assert_almost_equal(data, expected)



class TestComboFeatures(unittest.TestCase):
    def setUp(self):
        self.data = make_data(10)

    def test_dim_reduction(self):
        decomposer = decomposition.PCA(n_components=2)
        f = combo.DimensionReduction(['a', 'b', 'c'],
                decomposer=decomposer)
        data, ff = build_feature_safe(f, self.data)
        self.assertEqual(data.shape, (len(self.data), 2))
        decomposer = decomposition.PCA(n_components=2)
        expected = decomposer.fit_transform(self.data[['a', 'b', 'c']])
        assert_almost_equal(expected, data.values)


if __name__ == '__main__':
    unittest.main()

