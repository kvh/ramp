from utils import make_folds, _pprint
from pandas import Series, concat
import random
import hashlib
import copy
import numpy as np
from sklearn import cross_validation, ensemble, linear_model


class Selector(object):
    def __repr__(self):
        return '%s(%s)'%(self.__class__.__name__, _pprint(self.__dict__))

    @property
    def storable_hash(self):
        return repr(self)

    #@store
    def sets_config(self, ds, config):
        x = ds.get_train_x(config.features)
        y = ds.get_train_y(config.target)
        return self.sets(x, y)


class RandomForestSelector(Selector):

    def __init__(self, n=100, thresh=None, min_=True, classifier=False, seed=2345):
        self.n = n
        self.min = min_
        self.thresh = thresh
        self.seed = seed
        self.classifier = classifier

    def sets(self, x, y):
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

    def sets(self, x, y):
        alphas, active, coef_path = linear_model.lars_path(x.values, y.values)
        sets = []
        seen = set()
        print coef_path
        for coefs in coef_path.T:
            cols = [x.columns[i] for i in range(len(coefs)) if coefs[i] > 1e-9]
            sets.append(cols)
            for col in cols:
                if col not in seen:
                    print len(seen), col
                    seen.add(col)
        return sets
