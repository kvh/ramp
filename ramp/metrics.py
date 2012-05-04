import math
import numpy as np

class Metric(object):
    """
    Implements evaluate method that takes two
    Series (or DataFrames) and outputs a metric
    """
    reverse = False
    @property
    def name(self):
        return self.__class__.__name__.lower()

    def score(self, actual, predicted):
        raise NotImplementedError


class RMSE(Metric):
    def score(self, actual, predicted):
        return sum((actual - predicted)**2)/float(len(actual))

class LogLoss(Metric):
    def score(self, actual, predicted):
        return - sum(actual * np.log(predicted) + (1 - actual) * np.log(1 -
            predicted))/len(actual)
