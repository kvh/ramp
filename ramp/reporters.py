from sklearn import metrics
import numpy as np
from pandas import DataFrame

class Reporter(object):

    def set_config(self, config):
        self.config = config

    def update_with_model(self, model):
        pass

    def update_with_predictions(self, actuals, predictions):
        pass


class ModelOutliers(Reporter):
    pass


class ConfusionMatrix(Reporter):

    def update_with_predictions(self, actuals, predictions):
        cm = metrics.confusion_matrix(actuals, predictions)
        if hasattr(self.config.target, 'factors'):
            names = [f[0] for f in self.config.target.factors]
            df = DataFrame(cm, columns=names, index=names)
            print df.to_string()
        else:
            print cm

