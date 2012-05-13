import sys
sys.path.append('../..')
from ramp import *
from ramp import models
from ramp.metrics import *
from ramp import core
import unittest
import pandas
from pandas import Series, DataFrame, Index
# from datasets import *
# from  import core
# from features import *
# from models import *
# from metrics import *
from sklearn import linear_model
import numpy as np
import os, sys, tempfile
import random
# from test_features import make_data
from pandas.util.testing import assert_almost_equal

lmr = linear_model.LinearRegression()

class DummyPredictor(object):
    def __init__(self):
        pass

    def fit(self, x, y):
        self.fitx = x
        self.fity = y

    def predict(self, x):
        self.predictx = x
        p = np.zeros(len(x))
        p[0] = 100
        return p

class TestPredictor(object):
    def __init__(self, estimator):
        self.estimator = estimator
        self.n_fit = 0
        self.n_predicted = 0

    def fit(self, x, y):
        self.fitx = x
        self.fity = y
        self.n_fit += 1
        self.estimator.fit(x, y)

    def predict(self, x):
        self.predictx = x
        self.n_predicted += 1
        return self.estimator.predict(x)

lm = TestPredictor(lmr)

class RandomPredictor(DummyPredictor):
    def predict(self, x):
        DummyPredictor.predict(self, x)
        return np.random.randn(len(x))


def make_data(n):
        data = pandas.DataFrame(np.random.randn(n, 3),
                columns=['a','b','c'],
                index=range(10, n+10))
        data['2a_plus_b'] = data['a'] * 2 + data['b']
        validation_index = Index(range(n-100, n))
        # make validation data drastically different relationship
        data['2a_plus_b_valid_diff'] = data['2a_plus_b']
        data['2a_plus_b_valid_diff'].ix[validation_index] = (data['a'] * 10 - 20 * data['b']).reindex(validation_index)
        return data, validation_index

class TestTrainedFeature(unittest.TestCase):
    def setUp(self):
        n = 1000
        self.n = n
        self.data, validation_index = make_data(n)
        self.store = store.ShelfStore(tempfile.mkdtemp() + 'test.store')
        self.ds = DataSet(
                        name='$$test$$%d'%random.randint(100,10000),
                        data=self.data,
                        store=self.store,
                        validation_index=validation_index
        )

    def test_create_trained_feature(self):
        est = TestPredictor(lmr)
        f = Predictions(Configuration(
                features=['a'],
                target=F('2a_plus_b'),
                model=est)
                )
        r_before = repr(f)
        # train_index has not been set, should use all of training data
        res = f.create(self.ds)
        self.assertEqual(res.shape, (self.n,1))
        self.assertEqual(list(res.columns), ['Predictions[TestPredictor,1 features]() [01dd9cd6]'])
        self.assertEqual(est.fitx.shape, (len(self.ds.train_index),1))

        #assert_almost_equal(res, self.data['a'])
        self.assertEqual(len(self.store.get_store()), 3)
        r_after = repr(f)
        self.assertEqual(r_before, r_after)
        # recreate
        res2 = f.create(self.ds)
        assert_almost_equal(res, res2)
        self.assertEqual(len(self.store.get_store()), 3)

    def test_create_trained_feature2(self):
        est = TestPredictor(lmr)
        f = Predictions(Configuration(
                features=['a'],
                target=F('2a_plus_b'),
                model=est)
                )
        r_before = repr(f)
        # use subset for training
        tindex = self.ds.train_index[:100]
        res = f.create(self.ds, tindex)
        self.assertEqual(res.shape, (self.n,1))
        self.assertEqual(est.fitx.shape, (len(tindex),1))

        # use different subset for training
        tindex = self.ds.train_index[100:300]
        res2 = f.create(self.ds, tindex)
        self.assertTrue(len(res) == len(res2))

class DataSetTest(unittest.TestCase):

    def setUp(self):
        n = 1000
        # core.delete_data(force=True)
        self.data = pandas.DataFrame(np.random.randn(n, 3),
                columns=['a','b','c'],
                index=range(10, n+10))
        self.data['2a_plus_b'] = self.data['a'] * 2 + self.data['b']
        validation_index = Index(range(n-100, n))
        # make validation data drastically different relationship
        self.data['2a_plus_b_valid_diff'] = self.data['2a_plus_b']
        self.data['2a_plus_b_valid_diff'].ix[validation_index] = (self.data['a'] * 10 - 20 * self.data['b']).reindex(validation_index)
        self.store = store.ShelfStore(tempfile.mkdtemp() + 'test.store')
        self.ds = DataSet(
                        name='$$test-%d'%random.randint(99999, 999999),
                        data=self.data,
                        validation_index=validation_index,
                        store=self.store
                    )

    def make_config(self, **kwargs):
        conf = Configuration(
                features=['a', 'b'],
                target='2a_plus_b',
                metric=RMSE(),
                model=lm
                )
        conf.update(kwargs)
        return conf


    def test_basic_cv(self):
        c = self.make_config()
        n_folds = random.randint(2,10)
        repeat = 2
        scores = models.cv(self.ds, c, n_folds, repeat)
        self.assertAlmostEqual(scores.mean(), 0)
        self.assertEqual(len(scores), n_folds*repeat)

    def test_basic_cv_no_relationship(self):
        c = self.make_config(features=[
            'c'
            ])
        n_folds = random.randint(2,10)
        repeat = 2
        scores = models.cv(self.ds, c, n_folds, repeat)
        self.assertTrue(3 < scores.mean() < 6)
        self.assertEqual(len(scores), n_folds*repeat)

    def test_basic_cv_bad_valid(self):
        # shouldnt matter
        c = self.make_config(target='2a_plus_b_valid_diff')
        n_folds = random.randint(2,10)
        repeat = 2
        scores = models.cv(self.ds, c, n_folds, repeat)
        self.assertAlmostEqual(scores.mean(), 0)
        self.assertEqual(len(scores), n_folds*repeat)

    def test_basic_predict_relationship(self):
        c = self.make_config(target='2a_plus_b')
        preds = models.predict(self.ds, c, self.ds.validation_index,
                self.ds.train_index)
        mse = ((self.ds._data['2a_plus_b'].ix[self.ds.validation_index] -
                preds) ** 2).mean()
        print mse
        self.assertAlmostEqual(mse, 0)

    def test_basic_predict_bad_valid(self):
        c = self.make_config(target='2a_plus_b_valid_diff')
        preds = models.predict(self.ds, c, self.ds.validation_index,
                self.ds.train_index)
        mse = ((self.ds._data['2a_plus_b_valid_diff'].ix[self.ds.validation_index] -
                preds) ** 2).mean()
        print mse
        self.assertTrue( 800 > mse > 200)

    def test_blending(self):
        pcnf = Configuration(
            target='b',
            features=[
                '2a_plus_b_valid_diff',
                'a'],
            model= TestPredictor(linear_model.LinearRegression())
        )
        c = self.make_config(
                target='b',
                features=[
                    Predictions(pcnf)
                    ])
        scores = models.cv(self.ds, c)
        self.assertEqual(pcnf.model.n_fit, 5)
        print scores
        self.assertTrue( 3 < scores.mean() < 5)


if __name__ == '__main__':
    unittest.main()


