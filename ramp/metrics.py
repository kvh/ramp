import math
import numpy as np
from sklearn import metrics
# sklearn api change:
try:
    auc_scorer = metrics.auc_score
except AttributeError:
    auc_scorer = metrics.auc


class Metric(object):
    """
    Implements evaluate method that takes two
    Series (or DataFrames) and outputs a metric
    """
    # lower values are better by default, set reverse to true for
    # "bigger is better" metrics
    reverse = False
    @property
    def name(self):
        return self.__class__.__name__.lower()

    def score(self, actual, predicted):
        raise NotImplementedError

class SKLearnMetric(Metric):
    metric = None
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def score(self, actual, predicted):
        return self.metric(actual, predicted, **self.kwargs)


# Regression
class RMSE(Metric):
    def score(self, actual, predicted):
        return sum((actual - predicted)**2)/float(len(actual))


# Classification
class AUC(SKLearnMetric):
    reverse = True
    metric = staticmethod(auc_scorer)


class F1(SKLearnMetric):
    reverse = True
    metric = staticmethod(metrics.f1_score)


class HingeLoss(SKLearnMetric):
    metric = staticmethod(metrics.hinge_loss)


class LogLoss(Metric):
    def score(self, actual, predicted):
        return - sum(actual * np.log(predicted) + (1 - actual) * np.log(1 -
            predicted))/len(actual)


class GeneralizedMCC(Metric):
    """ Matthew's Correlation Coefficient generalized to multi-class case """
    def cov(self, c, n, flip=False):
        s = 0
        for k in range(n):
            if flip:
                s1 = sum([c[l,k] for l in range(n)])
                s2 = sum([c[g,f] for g in range(n) for f in range(n) if f != k])
            else:
                s1 = sum([c[k,l] for l in range(n)])
                s2 = sum([c[f,g] for g in range(n) for f in range(n) if f != k])
            s += s1 * s2
        return s

    def score(self, actual, predicted):
        c = metrics.confusion_matrix(actual, predicted)
        n = c.shape[0]
        numer = sum([c[k,k] * c[m,l] - c[l,k] * c[k,m] for k in range(n) for l in range(n) for m in range(n)])
        denom = math.sqrt(self.cov(c, n)) * math.sqrt(self.cov(c, n, flip=True))
        if abs(denom) < .00000001:
            return numer
        return numer/denom

