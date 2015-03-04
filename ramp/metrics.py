  # -*- coding: utf-8 -*-
'''
Metrics
-------

Estimator performance assessment metrics, both custom and imported from the
sklearn library of metrics. Sklearn metric documentation can be found at
http://scikit-learn.org/stable/modules/model_evaluation.html

Custom metrics/sklearn metrics can be generated/used by subclassing the
Metric/SKLearnMetric classes

'''

import logging
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
    Implements evaluate method that takes a Result object and outputs a score.
    """
    # lower values are better by default, set reverse to true for
    # "bigger is better" metrics
    reverse = False

    @property
    def name(self):
        return self.__class__.__name__.lower()

    def score(self, result):
        raise NotImplementedError


class SKLearnMetric(Metric):
    '''SKLearn library metric'''

    metric = None

    def __init__(self, metric, **kwargs):
        self.metric = metric
        self.kwargs = kwargs

    @property
    def name(self):
        return self.metric.__name__

    def score(self, result):
        return self.metric(result.y_test.values, result.y_preds, **self.kwargs)


def as_ramp_metric(metric_like):
    if isinstance(metric_like, basestring):
        if metric_like in sklearn_metric_lookup:
            return SKLearnMetric(sklearn_metric_lookup[metric_like])
        elif metric_like in metric_lookup:
            return metric_lookup[metric_like]()
        else:
            raise ValueError("Unrecognized metric: %s" % metric_like)
    if isinstance(metric_like, Metric):
        return metric_like
    if hasattr(metric_like, 'score'):
        return SKLearnMetric(metric_like)
    return metric_like()


# Regression
def rmse(*args, **kwargs):
    return np.sqrt(metrics.mean_squared_error(*args, **kwargs))


# Classification
class LogLoss(Metric):
    '''
    Logarithmic Loss: Logarithm of the likelihood function for a Bernoulli
    random distribution. https://www.kaggle.com/wiki/LogarithmicLoss
    '''
    def score(self, result):
        return - sum(result.y_test * np.log(result.y_preds) + (1 - actual) * np.log(1 -
            result.y_preds))/len(result.y_loss)


class MCC(SKLearnMetric):
    """
    Matthew's Correlation Coefficient.
    """
    reverse = True
    metric = staticmethod(metrics.matthews_corrcoef)


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

    def score(self, result):
        c = metrics.confusion_matrix(result.y_test, result.y_preds)
        n = c.shape[0]
        numer = sum([c[k,k] * c[m,l] - c[l,k] * c[k,m] for k in range(n) for l in range(n) for m in range(n)])
        denom = math.sqrt(self.cov(c, n)) * math.sqrt(self.cov(c, n, flip=True))
        if abs(denom) < .00000001:
            return numer
        return numer/denom


class ThresholdMetric(Metric):

    def __init__(self, threshold=None):
        self.threshold = threshold

    def score(self, result, threshold=None):
        if threshold is None:
            if self.threshold is None:
                raise ValueError("Threshold not specified")
            threshold = self.threshold
        try:
            return self.score_with_threshold(result, threshold)
        except ZeroDivisionError:
            return None

    def tp(self, result, threshold):
        return float(result.y_test[result.y_preds >= threshold].sum())

    def tn(self, result, threshold):
        return float((result.y_preds < threshold).sum()
                        - result.y_test[result.y_preds < threshold].sum())

    def fp(self, result, threshold):
        return float((result.y_preds >= threshold).sum()
                    - result.y_test[result.y_preds >= threshold].sum())

    def fn(self, result, threshold):
        return float(result.y_test[result.y_preds < threshold].sum())


class Recall(ThresholdMetric):
    """
    Recall: True positives / (True positives + False negatives)
    """

    def score_with_threshold(self, result, threshold):
        return self.tp(result, threshold) / result.y_test.sum()


class WeightedRecall(ThresholdMetric):
    # TODO: make a general WeightedThresholdMetric
    """
    Recall: Sum of weight column @ true positives  / sum of weight column @ (True positives + False negatives)
    """
    def __init__(self, threshold=None, weight_column=None):
        self.weight_column = weight_column
        super(WeightedRecall, self).__init__(threshold)

    def score_with_threshold(self, result, threshold):
        positive_indices = result.y_test[result.y_preds >= threshold].index
        return (result.original_data.loc[positive_indices][self.weight_column].sum() /
               float(result.original_data.loc[result.y_test.index][self.weight_column].sum()))


class PositiveRate(ThresholdMetric):
    """
    Positive rate: (True positives + False positives) / Total count
    """

    def score_with_threshold(self, result, threshold):
        return ((self.tp(result, threshold) + self.fp(result, threshold)) /
                                 result.y_test.count())


class Precision(ThresholdMetric):
    """
    Precision: True positives / (True positives + False positives)
    """

    def score_with_threshold(self, result, threshold):
        return self.tp(result, threshold) / (self.tp(result, threshold) +
                                                self.fp(result, threshold))


class FalsePositiveRate(ThresholdMetric):
    """
    Precision: False positives / (False positives + True negatives)
    """

    def score_with_threshold(self, result, threshold):
        return self.fp(result, threshold) / (self.tn(result, threshold) +
                                                self.fp(result, threshold))


sklearn_metric_lookup = {
    # regression
    'mse': metrics.mean_squared_error,
    'rmse': rmse,
    'r2': metrics.r2_score,
    'mae': metrics.mean_absolute_error,
    'explained_variance': metrics.explained_variance_score,

    # classification
    'mcc': metrics.matthews_corrcoef,
    'f1': metrics.f1_score,
    'auc': auc_scorer,
}
metric_lookup = {
    'precision': Precision,
    'positive_rate': PositiveRate,
    'percentile': PositiveRate,
    'recall': Recall,
    'fpr':FalsePositiveRate,
    'fall_out':FalsePositiveRate,
}

