import sys
sys.path.append('../..')
import unittest

import numpy as np
import pandas as pd
from pandas import DataFrame, Series, Index

from ramp.features.base import F, Map
from ramp.folds import *
from ramp.utils import *


class TestFolds(unittest.TestCase):

    def test_balanced_folds(self):
        n = 100000
        r = 4
        n_folds = 4
        df = pd.DataFrame({'a':np.arange(n), 'y':np.hstack([np.ones(n/r), np.zeros(n/r * (r -1))])})
        balanced_folds = BalancedFolds(n_folds, F('y'), df, seed=1)
        folds = list(balanced_folds)
        self.assertEqual(len(folds), n_folds)
        te = pd.Index([])
        for train, test in folds:
            self.assertFalse(train & test)
            self.assertFalse(te & test)
            te = te | test
            train_y = df.y[train]
            test_y = df.y[test]
            # ensure postive ratios are correct
            pos_ratio = sum(train_y) / float(len(train_y))
            self.assertAlmostEqual(pos_ratio, 1. / r)
            pos_ratio = sum(test_y) / float(len(test_y))
            self.assertAlmostEqual(pos_ratio, 1. / r)
        # ensure all instances were used in test
        self.assertEqual(len(te), n)

    def test_bootstrapped_folds(self):
        n = 10000
        r = 4
        n_folds = 10
        ptr = 2000
        pte = 500
        ntr = 6000
        nte = 1000
        df = pd.DataFrame({'a':np.arange(n), 'y':np.hstack([np.ones(n/r), np.zeros(n/r * (r -1))])})
        balanced_folds = BootstrapFolds(n_folds, F('y'), df, seed=1,
                                        pos_train=ptr, neg_train=ntr, pos_test=pte, neg_test=nte)
        folds = list(balanced_folds)
        self.assertEqual(len(folds), n_folds)
        te = set()
        for train, test in folds:
            self.assertFalse(set(train) & set(test))
            te = te | set(train)
            train_y = df.y[train]
            test_y = df.y[test]
            # ensure sizes are correct
            self.assertEqual(len(train_y), ptr + ntr)
            self.assertEqual(len(test_y), pte + nte)
            self.assertEqual(sum(train_y), ptr)
            self.assertEqual(sum(test_y), pte)
        # ensure all instances were used in test (well, almost)
        self.assertEqual(len(te), 9998) # seed is set

    def test_basic_folds(self):
        n = 10000
        n_folds = 10
        df = pd.DataFrame({'a':np.arange(n)})
        bfolds = BasicFolds(n_folds, df)
        folds = list(bfolds)
        self.assertEqual(len(folds), n_folds)
        te = set()
        for train, test in folds:
            self.assertFalse(set(train) & set(test))
            te = te | set(test)
            train_y = df.loc[train]
            test_y = df.loc[test]
            # ensure sizes are correct
            self.assertEqual(len(train_y), n / float(n_folds) * 9)
            self.assertEqual(len(test_y), n / float(n_folds))
        # ensure all instances were used in test
        self.assertEqual(len(te), n)


if __name__ == '__main__':
    unittest.main(verbosity=2)

