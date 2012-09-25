from utils import make_folds, _pprint
from pandas import Series, concat
from scipy.stats import norm
import random
import hashlib
import copy
import numpy as np
from sklearn import cross_validation, ensemble, linear_model
from utils import get_hash

class Selector(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __repr__(self):
        return '%s(%s)'%(self.__class__.__name__, _pprint(self.__dict__))

    def sets_config(self, ds, config, index):
        try:
            return ds.store.load('%r-%r-%s'%(config.features, config.target,
                get_hash(index)))
        except KeyError:
            pass
        x = ds.get_train_x(config.features)
        y = ds.get_train_y(config.target)
        sets = self.sets(x, y)
        ds.store.save('%r-%r'%(config.features, config.target), sets)
        return sets


class RandomForestSelector(Selector):

    def __init__(self, n=100, thresh=None, min_=True, classifier=False,
            seed=2345, *args, **kwargs):
        self.n = n
        self.min = min_
        self.thresh = thresh
        self.seed = seed
        self.classifier = classifier
        super(RandomForestSelector, self).__init__(*args, **kwargs)

    def sets(self, x, y, n_keep):
        cls = ensemble.RandomForestRegressor
        if self.classifier:
            cls = ensemble.RandomForestClassifier
        rf = cls(n_estimators=self.n,
                compute_importances=True,
                random_state=self.seed,
                n_jobs=-1)
        rf.fit(x.values, y.values)
        importances = rf.feature_importances_
        imps = sorted(zip(importances, x.columns),
                reverse=True)
        if self.verbose:
            for i, x in enumerate(imps):
                imp, f = x
                print '%d\t%0.4f\t%s'%(i,imp, f)
        if self.thresh:
            imps = [t for t in imps if t[0] > self.thresh]
        return [t[1] for t in imps[:n_keep]]
#        sets = [[t[1] for t in imps[:i+1]] for i in range(len(imps))]
#        return sets

    def sets_cv(self, x, y):
        totals = [0]*len(x.columns)
        if self.min:
            totals = [1000] * len(x.columns)
        i = 0
        for train, test in cross_validation.KFold(n=len(y), k=4):
            i += 1
            print "RF selector computing importances for fold", i
            cls = ensemble.RandomForestRegressor
            if self.classifier:
                cls = ensemble.RandomForestClassifier
            rf = cls(n_estimators=self.n,
                    compute_importances=True,
                    random_state=self.seed,
                    n_jobs=-1)
            rf.fit(x.values[train], y.values[train])
            importances = rf.feature_importances_
            if self.min:
                totals = [min(imp, t) for imp, t in zip(importances, totals)]
            else:
                totals = [imp + t for imp, t in zip(importances, totals)]
        imps = sorted(zip(totals, x.columns),
                reverse=True)
        for i, x in enumerate(imps):
            imp, f = x
            print '%d\t%0.4f\t%s'%(i,imp, f)
        if self.thresh:
            imps = [t for t in imps if t[0] > self.thresh]
        sets = [[t[1] for t in imps[:i+1]] for i in range(len(imps))]
        return sets

class StepwiseForwardSelector(Selector):
    def __init__(self, n=100, min_=True):
        self.n = n
        self.min = min_

    def sets(self, x, y):
        lm = linear_model.LinearRegression(normalize=True)
        remaining = x.columns
        curr = []
        print "stepwise forward"
        for i in range(self.n):
            if i % 10 == 0:
                print i, 'features'
            coefs = []
            for col in remaining:
                cols = curr + [col]
                fcf = 1e12
                for train, test in cross_validation.KFold(n=len(y), k=4):
                    # computes fits over folds, uses lowest computed
                    # coefficient as value for comparison
                    lm.fit(x[cols].values[train], y.values[train])
                    cf = lm.coef_[-1]
                    if np.isnan(x[col].std()) or x[col].std() < 1e-7:
                        cf = 0
                    cf = abs(cf)
                    fcf = min(cf, fcf)
                coefs.append(fcf)
            coef, col = max(zip(coefs, remaining))
            print "adding column", col
            curr.append(col)
            remaining = remaining.drop([col])
            yield list(curr)

class LassoPathSelector(Selector):

    def sets(self, x, y, n_keep):
        alphas, active, coef_path = linear_model.lars_path(x.values, y.values)
        sets = []
        seen = set()
        print coef_path
        for coefs in coef_path.T:
            cols = [x.columns[i] for i in range(len(coefs)) if coefs[i] > 1e-9]
            if len(cols) >= n_keep:
                return cols
        return cols
#            sets.append(cols)
#            for col in cols:
#                if col not in seen:
#                    print len(seen), col
#                    seen.add(col)
#        return sets


class BinaryFeatureSelector(Selector):
    """ Only for binary classification and binary(-able) features """

    def __init__(self, type='bns', *args, **kwargs):
        """ type in ('bns', 'acc') 
        see: jmlr.csail.mit.edu/papers/volume3/forman03a/forman03a.pdf"""
        self.type = type
        super(BinaryFeatureSelector, self).__init__(*args, **kwargs)

    def sets(self, x, y, n_keep):
        cnts = y.value_counts()
        assert(len(cnts) == 2)
        print "Computing binary feature scores for %d features..." % len(x.columns)
        scores = []
        for c in x.columns:
            true_positives = np.count_nonzero(np.logical_and(x[c], y))
            false_positives = np.count_nonzero(np.logical_and(x[c], np.logical_not(y)))
            tpr = max(0.0005, true_positives / float(cnts[1]))
            fpr = max(0.0005, false_positives / float(cnts[0]))
            tpr = min(.9995, tpr)
            fpr = min(.9995, fpr)
            if self.type == 'bns':
                score = abs(norm.ppf(tpr) - norm.ppf(fpr))
            elif self.type == 'acc':
                score = abs(tpr - fpr)
            scores.append((score, c))
        scores.sort(reverse=True)
        if self.verbose:
            # just show top few hundred
            print scores[:200]
        return [s[1] for s in scores[:n_keep]]

class InformationGainSelector(Selector):
    """ Only for binary classification """

    def sets(self, x, y, n_keep):
        cnts = y.value_counts()
        assert(len(cnts) == 2)
        print "Computing IG scores..."
        scores = []
        for c in x.columns:
            true_positives = sum(np.logical_and(x[c], y))
            false_positives = sum(np.logical_and(x[c], np.logical_not(y)))
            score = abs(norm.ppf(true_positives / float(cnts[1])) - norm.ppf(false_positives / float(cnts[0])))
            scores.append((score, c))
        scores.sort(reverse=True)
        return [s[1] for s in scores[:n_keep]]
