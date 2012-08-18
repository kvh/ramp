import sys
sys.path.append('../..')
from ramp import *
from ramp import core
from ramp import store
import unittest
import pandas
import tempfile

from sklearn import linear_model
import numpy as np
import os, sys

from pandas.util.testing import assert_almost_equal
from test_models import lm


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

    def test_basefeature(self):
        f = BaseFeature('col1')
        self.assertEqual(f.feature, 'col1')
        self.assertEqual(str(f), 'col1')
        self.assertEqual(repr(f), "'col1'")
        self.assertEqual(f.unique_name, 'col1')
        self.assertFalse(f.is_trained())

    def test_constantfeature_int(self):
        f = ConstantFeature(1)
        self.assertEqual(f.feature, 1)
        self.assertEqual(str(f), '1')
        self.assertEqual(repr(f), '1')
        self.assertEqual(f.unique_name, '1')
        self.assertFalse(f.is_trained())

    def test_constantfeature_float(self):
        f = ConstantFeature(math.e)
        e_str = '2.71828182846'
        self.assertEqual(f.feature, math.e)
        self.assertEqual(str(f), e_str)
        self.assertEqual(repr(f), '2.718281828459045')
        self.assertEqual(f.unique_name, e_str)
        self.assertFalse(f.is_trained())

    def test_combofeature(self):
        f = ComboFeature(['col1', 'col2'])
        for sf in f.features:
            self.assertIsInstance(sf, BaseFeature)
        self.assertEqual(str(f), 'Combo(col1, col2)')
        self.assertEqual(f.unique_name, 'Combo(col1, col2) [201b1e5d]')
        self.assertEqual(repr(f), "ComboFeature(_name='Combo',"
        "features=['col1', 'col2'])")
        self.assertFalse(f.is_trained())

    def test_feature(self):
        f = Feature('col1')
        self.assertIsInstance(f.feature, BaseFeature)
        self.assertEqual(str(f), 'col1')
        self.assertEqual(f.unique_name, 'col1 [4e89804a]')
        self.assertEqual(repr(f), "Feature(_name='',"
        "feature='col1',features=['col1'])")
        self.assertFalse(f.is_trained())



class TestFeatureCreate(unittest.TestCase):
    def setUp(self):
        n = 100
        self.n = n
        self.data = make_data(n)
        #core.delete_data(force=True)
        self.store = store.ShelfStore(tempfile.mkdtemp() + 'test.store')
        self.ds = DataSet(
                        name='$$test$$%d'%random.randint(100,10000),
                        data=self.data,
                        store=self.store,)

    def test_feature_save(self):
        f = Feature('col1')
        f.dataset = self.ds
        k = 'test'
        v = 123
        f.save(k, v)
        self.assertEqual(f.load(k), v)
        f2 = Feature('col2')
        f2.dataset = self.ds
        v2 = 456
        f2.save(k, v2)
        self.assertEqual(f.load(k), v)
        self.assertEqual(f2.load(k), v2)

    def test_create_feature(self):
        f = Feature('a')
        r_before = repr(f)
        res = f.create(self.ds)
        self.assertEqual(res.shape, (self.n,1))
        self.assertEqual(res.columns, ['a [4c205f6a]'])
        assert_almost_equal(res['a [4c205f6a]'], self.data['a'])
        self.assertEqual(len(self.store.get_store()), 1)
        r_after = repr(f)
        self.assertEqual(r_before, r_after)
        # recreate
        f = Feature('a')
        res = f.create(self.ds)
        self.assertEqual(len(self.store.get_store()), 1)





if __name__ == '__main__':
    unittest.main()


