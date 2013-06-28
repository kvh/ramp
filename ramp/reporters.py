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


class RFImportance(ramp.reporters.Reporter):
    def __init__(self):
        self.importances = []

    def update_with_model(self, model):
        try:
            imps = model.feature_importances_
        except AttributeError:
            print "Warning: Model has no feature importances"
            return
        print "Feature Importances"
        print "Rank\tGini\tFeature"
        imps = sorted(zip(imps, model.column_names),
                reverse=True)
        for i, x in enumerate(imps):
            imp, f = x
            print '%d\t%0.4f\t%s'%(i+1,imp, f)
        self.importances.append(imps)
