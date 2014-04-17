import pandas as pd
import numpy as np
import random
from builders import build_target_safe

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
        y, ff= build_target_safe(self.target, self.data)
        self.y = y
        self.negatives = y[~y.astype('bool')].index
        self.positives = y[y.astype('bool')].index

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

    def __init__(self, num_folds, target, data, seed=None,
                  pos_train=None, pos_test=None, neg_train=None, neg_test=None):
        super(BootstrapFolds, self).__init__(num_folds, target, data, seed)
        if (any([pos_train, pos_test, neg_train, neg_test])
                and not all([pos_train, pos_test, neg_train, neg_test])):
            raise ValueError("Please specify all four sizes, or none at all")
        self.pos_train = pos_train
        self.neg_train = neg_train
        self.pos_test = pos_test
        self.neg_test = neg_test

    def compute_folds(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        folds = []
        for i in range(self.num_folds):
            train_neg = pd.Index(np.random.choice(self.negatives, self.neg_train, replace=True))
            test_neg = pd.Index(np.random.choice(self.negatives - train_neg, self.neg_test, replace=True))
            train_pos = pd.Index(np.random.choice(self.positives, self.pos_train, replace=True))
            test_pos = pd.Index(np.random.choice(self.positives - train_pos, self.pos_test, replace=True))
            fold = (train_neg.append(train_pos), test_neg.append(test_pos))
            folds.append(fold)
        self.folds = folds


make_default_folds = BasicFolds