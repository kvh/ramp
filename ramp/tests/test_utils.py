import sys
sys.path.append('../..')
import unittest

import numpy as np
import pandas as pd
from pandas import DataFrame, Series, Index
from pandas.util.testing import assert_frame_equal, assert_index_equal

from ramp.features.base import F, Map
from ramp.utils import *


class TestUtils(unittest.TestCase):

    def test_shuffle_df(self):
        n = 100
        df = pd.DataFrame({'a':np.random.rand(n), 
                           'b':np.random.rand(n)})
        df_shuffled = shuffle_df(df)
        self.assertEqual(df.shape, df_shuffled.shape)
        self.assertNotEqual(tuple(df['a']), tuple(df_shuffled['a']))
        self.assertNotEqual(tuple(df['b']), tuple(df_shuffled['b']))
        self.assertNotEqual(tuple(df.index), tuple(df_shuffled.index))
        assert_frame_equal(df.sort(), df_shuffled.sort())

    def test_np_hashes(self):
        a = np.random.randn(20)
        h = get_np_hash(a)
        a[0] = 200000
        h2 = get_np_hash(a)
        self.assertNotEqual(h, h2)
        b = a[0]
        a[0] = b
        self.assertEqual(h2, get_np_hash(a))

    def test_stable_repr(self):
        f = F('test')
        f2 = F('test')
        # equalivalent objects should have same repr
        self.assertEqual(stable_repr(f), stable_repr(f2))

        # repr is independent of object ids
        class Test: pass
        f.f = Test()
        r1 = stable_repr(f)
        f.f = Test()
        r2 = stable_repr(f)
        self.assertEqual(r1, r2)


if __name__ == '__main__':
    unittest.main(verbosity=2)

