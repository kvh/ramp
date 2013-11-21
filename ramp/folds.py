import pandas as pd
import numpy as np
import random
from builders import build_target


class WeightedSampleFolds(object):
    
    def __init__(self, folds, positive_proportion_test, positive_ratio_test,
            positive_ratio_train=None, verbose=True):
        self.folds = folds
        self.positive_proportion_test = positive_proportion_test
        self.positive_ratio_test = positive_ratio_test
        self.positive_ratio_train = positive_ratio_train
        self.verbose = verbose

    def set_context(self, config, context):
        self.context = context
        self.config = config

    def __iter__(self):
        for i in range(self.folds):
            y = build_target(self.config.target, self.context)
            positives = y[y != 0].index
            negatives = y[y == 0].index
            np = len(positives)
            print "posssss", np, len(y)
            nn = len(negatives)
            test_positives = random.sample(positives, int(np * self.positive_proportion_test))
            np_test = len(test_positives)
            test_negatives = random.sample(negatives, int(np_test * (1 / self.positive_ratio_test  - 1)))
            nn_test = len(test_negatives)
            test = test_positives + test_negatives
            if self.positive_ratio_train:
                train_negs = random.sample(negatives - test_negatives, int((np - np_test) * (1 / self.positive_ratio_train - 1)))
                train = train_negs + list(positives - test_positives)
                nn_train = len(train_negs)
            else:
                train = y.index - test
                nn_train = nn - nn_test
            if self.verbose:
                print "Weighted Sample Folds:"
                print "\tPos\tNeg\tPos pct"
                print "Train:\t%d\t%d\t%0.3f" % (np - np_test, nn_train, (np - np_test) / float( np - np_test + nn_train))
                print "Test:\t%d\t%d\t%0.3f" % (np_test, nn_test, np_test / float(nn_test + np_test))
            yield pd.Index(train), pd.Index(test)


class SampledFolds(object):
    
    def __init__(self, folds, pos_train, neg_train, pos_test, neg_test,
            verbose=True):
        self.folds = folds
        self.pos_train = pos_train
        self.neg_train = neg_train
        self.pos_test = pos_test
        self.neg_test = neg_test
        self.verbose = verbose

    def set_context(self, config, context):
        self.context = context
        self.config = config

    def __iter__(self):
        for i in range(self.folds):
            y = build_target(self.config.target, self.context)
            positives = y[y != 0].index
            negatives = y[y == 0].index
            np = len(positives)
            nn = len(negatives)
            test_positives = random.sample(positives, self.pos_test)
            test_negatives = random.sample(negatives, self.neg_test)
            train_positives = random.sample(positives - test_positives, self.pos_train)
            train_negatives = random.sample(negatives - test_negatives, self.neg_train)
            test = test_positives + test_negatives
            train = train_positives + train_negatives
            if self.verbose:
                print "Sampled Folds:"
                print "\tPos\tNeg\tPos pct"
                print "Train:\t%d\t%d\t%0.3f" % (self.pos_train, self.neg_train,
                        self.pos_train / float( self.pos_train + self.neg_train))
                print "Test:\t%d\t%d\t%0.3f" % (self.pos_test, self.neg_test, self.pos_test / float(self.neg_test + self.pos_test))
            yield pd.Index(train), pd.Index(test)


class SequenceFolds(object):
    
    def __init__(self, train_size, test_size,
            verbose=True):
        self.train_size = train_size
        self.test_size = test_size
        self.verbose = verbose

    def set_context(self, config, context):
        self.context = context
        self.config = config

    def __iter__(self):
        index = self.context.data.index
        for i in range(self.train_size, len(index) - self.test_size):
            train = index[:i]
            test = index[i:i+self.test_size]
            yield pd.Index(train), pd.Index(test)
        
