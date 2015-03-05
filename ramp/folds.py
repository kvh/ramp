import logging
import random

import numpy as np
import pandas as pd

from ramp.builders import build_target_safe

#TODO: how to repeat folds?


class BasicFolds(object):
    def __init__(self, num_folds, data, repeat=1, seed=None):
        self.num_folds = num_folds
        self.data = data
        self.seed = seed
        self.repeat = repeat

    def __iter__(self):
        n = len(self.data)
        index = self.data.index
        indices = range(n)
        foldsize = n / self.num_folds
        folds = []
        if self.seed is not None:
            np.random.seed(self.seed)
        for i in range(self.repeat):
            np.random.shuffle(indices)
            for i in range(self.num_folds):
                test = index[indices[i*foldsize:(i + 1)*foldsize]]
                train = index - test
                assert not (train & test)
                fold = (pd.Index(train), pd.Index(test))
                yield fold


class WatertightFolds(BasicFolds):
    """
    Ensure that there is no leakage across a particular factor, given by
    `leakage_column`.

    For example, if there are multiple entries for a given user, this could be
    used to ensure that each user is completely contained in a single fold.
    """
    def __init__(self, num_folds, data, leakage_column, **kwargs):
        super(WatertightFolds, self).__init__(num_folds, data, **kwargs)
        self.leakage_column = leakage_column

    def __iter__(self):
        n = len(self.data)
        index = self.data.index
        indices = range(n)
        foldsize = n / self.num_folds
        folds = []

        if self.seed is not None:
            np.random.seed(self.seed)

        for i in range(self.repeat):
            watertight_bins = self.data.groupby(self.leakage_column)[self.data.columns[0]].count()
            watertight_bins = watertight_bins.reindex(np.random.permutation(watertight_bins.index))
            watertight_bins_cum = watertight_bins.cumsum()
            for i in range(self.num_folds):
                test_bins = watertight_bins_cum[(watertight_bins_cum >  i * foldsize) &
                                                (watertight_bins_cum <= (i+1) * foldsize)]
                test =index[self.data[self.leakage_column].isin(test_bins.index)]
                train = index - test

                # Sanity checks
                assert not (train & test)
                if np.abs(len(test) - foldsize) > 0.05*foldsize:
                    # Folds will not be exact in size, but a warning will be
                    # emitted if they are far from the expected value.
                    logging.warn("Fold deviated from expected size. Target: {target} Actual: {actual}".format(target=foldsize, actual=len(test)))
                fold = (pd.Index(train), pd.Index(test))
                yield fold


class BinaryTargetFolds(object):
    def __init__(self, target, data, seed=None):
        self.seed = seed
        self.target = target
        self.data = data
        self.folds = None
        self.y = None

    def compute_folds(self):
        raise NotImplementedError

    def build_target(self):
        y, ff = build_target_safe(self.target, self.data)
        self.y = y
        self.negatives = y[~y.astype('bool')].index
        self.positives = y[y.astype('bool')].index
        self.n_positives = len(self.positives)
        self.n_negatives = len(self.negatives)

    def randomize(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        neg = pd.Index(np.random.permutation(self.negatives))
        pos = pd.Index(np.random.permutation(self.positives))
        return neg, pos

    def __iter__(self):
        if self.y is None:
            self.build_target()
        if self.folds is None:
            self.compute_folds()
        for fold in self.folds:
            yield fold


class BalancedFolds(BinaryTargetFolds):
    def __init__(self, num_folds, target, data, seed=None):
        self.num_folds = num_folds
        super(BalancedFolds, self).__init__(target, data, seed)

    def compute_folds(self):
        neg, pos = self.randomize()
        nn = len(neg) / self.num_folds
        np = len(pos) / self.num_folds
        folds = []
        for i in range(self.num_folds):
            s = i * nn
            e = (i + 1) * nn
            train_neg, test_neg = neg[:s] + neg[e:], neg[s:e]
            s = i * np
            e = (i + 1) * np
            train_pos, test_pos = pos[:s] + pos[e:], pos[s:e]
            fold = (train_neg + train_pos, test_neg + test_pos)
            folds.append(fold)
        self.folds = folds


class BootstrapFolds(BalancedFolds):
    def __init__(self, num_folds, target, data,
                  pos_train=None, pos_test=None, neg_train=None, neg_test=None,
                  train_pos_percent=None, test_pos_percent=None, train_percent=None,
                  seed=None):
        super(BootstrapFolds, self).__init__(num_folds, target, data, seed)
        if (any([pos_train, pos_test, neg_train, neg_test])
                and not all([pos_train, pos_test, neg_train, neg_test])):
            raise ValueError("Please specify all four sizes, or none at all")
        if (any([train_pos_percent, test_pos_percent, train_percent])
                and not all([train_pos_percent, test_pos_percent, train_percent])):
            raise ValueError("Please specify both percentages, or none at all")
        if (train_pos_percent is None) == (pos_test is None):
            raise ValueError("Please specify either sizes or percentages, not both")
        self.pos_train = pos_train
        self.neg_train = neg_train
        self.pos_test = pos_test
        self.neg_test = neg_test
        self.train_pos_percent = train_pos_percent
        self.test_pos_percent = test_pos_percent
        self.train_percent = train_percent
        self.using_percents = False
        if self.test_pos_percent is not None:
            self.using_percents = True

    def from_sizes(self):
        train_neg = pd.Index(np.random.choice(self.negatives, self.neg_train, replace=True))
        test_neg = pd.Index(np.random.choice(self.negatives - train_neg, self.neg_test, replace=True))
        train_pos = pd.Index(np.random.choice(self.positives, self.pos_train, replace=True))
        test_pos = pd.Index(np.random.choice(self.positives - train_pos, self.pos_test, replace=True))
        return train_neg, test_neg, train_pos, test_pos

    def from_percents(self):
        # Assumes positive is constrained (rarer) class
        rp = self.train_percent * self.n_positives
        tp = (1 - self.train_percent) * self.n_positives
        rn = rp / self.train_pos_percent - rp
        tn = tp / self.test_pos_percent - tp
        train_neg = pd.Index(np.random.choice(self.negatives,
                                              int(rn),
                                              replace=True))
        test_neg = pd.Index(np.random.choice(self.negatives - train_neg,
                                             int(tn),
                                             replace=True))
        train_pos = pd.Index(np.random.choice(self.positives,
                                              int(rp),
                                              replace=True))
        test_pos = pd.Index(np.random.choice(self.positives - train_pos,
                                             int(tp),
                                             replace=True))
        return train_neg, test_neg, train_pos, test_pos

    def compute_folds(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        folds = []
        for i in range(self.num_folds):
            if self.using_percents:
                train_neg, test_neg, train_pos, test_pos = self.from_percents()
            else:
                train_neg, test_neg, train_pos, test_pos = self.from_sizes()
            fold = (train_neg.append(train_pos), test_neg.append(test_pos))
            folds.append(fold)
        self.folds = folds


class BootstrapFoldsByPercentPositive(BootstrapFolds):

    def __init__(self, num_folds, target, data,
                 train_pos_percent, test_pos_percent, train_percent,
                 seed=None):
        super(BootstrapFoldsByPercentPositive,
              self).__init__(num_folds, target, data, seed=seed,
                             train_pos_percent=train_pos_percent,
                             test_pos_percent=test_pos_percent,
                             train_percent=train_percent)

class BootstrapFoldsBySize(BootstrapFolds):

    def __init__(self, num_folds, target, data,
                 pos_train=None, pos_test=None, neg_train=None, neg_test=None,
                 seed=None):
        super(BootstrapFoldsBySize, self).__init__(num_folds, target, data, seed=seed,
                                                   pos_train=pos_train,
                                                   pos_test=pos_test,
                                                   neg_train=neg_train,
                                                   neg_test=neg_test)

make_default_folds = BasicFolds
