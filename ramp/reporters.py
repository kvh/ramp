from sklearn import metrics
import numpy as np
from pandas import DataFrame

class Reporter(object):

    def set_config(self, config):
        self.config = config

    def update_with_model(self, model):
        pass

    def update_with_predictions(self, context, x, actuals, predictions):
        pass


class ModelOutliers(Reporter):
    pass


class ConfusionMatrix(Reporter):

    def update_with_predictions(self, context, x, actuals, predictions):
        cm = metrics.confusion_matrix(actuals, predictions)
        self.config.target.context = context
        try:
            factors = self.config.target.get_prep_data()
        except KeyError:
            print cm
            return
        if factors:
            names = [f[0] for f in factors]
            df = DataFrame(cm, columns=names, index=names)
            print df.to_string()
        else:
            print cm


class MislabelInspector(Reporter):

    def __init__(self, reported_features=None):
        self.reported_features = reported_features or []

    def update_with_model(self, model):
        pass

    def update_with_predictions(self, context, x, actuals, predictions):
        for ind in actuals.index:
            a, p = actuals[ind], predictions[ind]
            if a != p:
                print "-" * 20
                print "Actual: %s\tPredicted: %s" % (a, p)
                print x.ix[ind]
                print context.data.ix[ind]
                i = raw_input()
                if i.startswith('c'):
                    break
