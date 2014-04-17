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

    def score(self, result):
        return self.metric(result.y_test, result.y_preds, **self.kwargs)


# Regression
class RMSE(Metric):
    '''Mean Squared Error: The average of the squares of the errors.'''
    def score(self):
        return sum((result.y_test - result.y_preds)**2)/float(len(result.y_test))


# Classification
class AUC(SKLearnMetric):
    '''
    Area Under the Curve (AUC): area under the reciever operating 
    characteristic (ROS curve)
    '''
    reverse = True
    metric = staticmethod(auc_scorer)


class F1(SKLearnMetric):
    '''F-measure: Weighted average of the precision and recall'''
    reverse = True
    metric = staticmethod(metrics.f1_score)


class HingeLoss(SKLearnMetric):
    '''Hinge Loss (non-regularized): Classifier loss function'''
    metric = staticmethod(metrics.hinge_loss)


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



class ArgMetric(Metric):
    """
    Implements an evaluate method that takes a Result object and an argument and
    returns a score.
    """
    def score(self, result, arg):
        raise NotImplementedError

class Recall(ArgMetric):
    """
    Recall: True positives / (True positives + False negatives)
    """
    def score(self, result, threshold):
        return result.y_test[result.y_preds > threshold].sum() / result.y_test.sum()

class WeightedRecall(ArgMetric):
    """
    Recall: Sum of weight column @ true positives  / sum of weight column @ (True positives + False negatives)
    """
    def __init__(self, weight_column):
        self.weight_column = weight_column

    def score(self, result, threshold):
        positive_indices = result.y_test[result.y_preds > threshold].index
        return result.original_data.loc[positive_indices][self.weight_column].sum() / \
               result.original_data.loc[result.y_test.index][self.weight_column].sum()

class PositiveRate(ArgMetric):
    """
    Positive rate: (True positives + False positives) / Total count
    """
    def score(self, result, threshold):
        return result.y_test[result.y_preds > threshold].count() / result.y_test.count()
