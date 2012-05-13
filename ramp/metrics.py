import math
import numpy as np
from sklearn import metrics

class Metric(object):
    """
    Implements evaluate method that takes two
    Series (or DataFrames) and outputs a metric
    """
    # smaller is better by default, set reverse to True for bigger is better
    # metrics
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
    metric = staticmethod(metrics.auc)

class F1(SKLearnMetric):
    reverse = True
    metric = staticmethod(metrics.f1_score)

class HingeLoss(SKLearnMetric):
    metric = staticmethod(metrics.hinge_loss)

class LogLoss(Metric):
    def score(self, actual, predicted):
        return - sum(actual * np.log(predicted) + (1 - actual) * np.log(1 -
            predicted))/len(actual)
