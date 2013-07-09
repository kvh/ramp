from sklearn import metrics
import numpy as np
from pandas import DataFrame
from collections import defaultdict


class Reporter(object):

    def reset(self):
        """
        Clear any state.
        """

    def set_config(self, config):
        self.config = config

    def update_with_model(self, model):
        pass

    def update_with_predictions(self, context, x, actuals, predictions):
        pass

    def report(self):
        """ Called at end of cross-validation run. Used for reporting
        aggregate information.
        """


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


class RFImportance(Reporter):
    def __init__(self):
        self.reset()

    def reset(self):
        self.importances = []

    def update_with_model(self, model):
        try:
            imps = model.feature_importances_
        except AttributeError:
            print "Warning: Model has no feature importances"
            return
        if imps is None:
            print "Warning: Model has no feature importances"
            return
        imps = sorted(zip(imps, model.column_names),
                reverse=True)
        print self.print_string(imps)
        self.importances.append(imps)
    
    def print_string(self, imps):
        s = "Average Feature Importances\n"
        s += "Rank\tGini\tFeature\n"
        for i, x in enumerate(imps):
            imp, f = x
            s += '%d\t%0.4f\t%s\n'%(i+1,np.average(imp), f)
        return s

    def report(self):
        if not self.importances:
            return
        d = defaultdict(list)
        for imps in self.importances:
            for imp, f in imps:
                d[f].append(imp)
        return self.print_string(sorted(zip(d.values(), d.keys()), key=lambda x: np.average(x[0]), reverse=True))


class PRCurve(Reporter):
    def update_with_predictions(self, context, x, actuals, predictions):
        p, r, t = metrics.precision_recall_curve(actuals, predictions)
        print zip(p, r)


class ROCCurve(Reporter):
    def __init__(self, show_plot=True):
        self.show_plot = show_plot

    def update_with_predictions(self, context, x, actuals, predictions):
        fpr, tpr, thresholds = metrics.roc_curve(actuals, predictions)
        print "ROC thresholds"
        print "FP Rate\tTP Rate"
        for x in zip(fpr, tpr):
            print '%0.4f\t%0.4f' % x
        if not self.show_plot:
            return
        try:
            import pylab as pl
        except ImportError as e:
            print "ERROR: You must install matplotlib in order to see plots"
            return
        pl.plot(fpr, tpr, label='ROC curve')
        pl.plot([0, 1], [0, 1], 'k--')
        pl.xlim([0.0, 1.0])
        pl.ylim([0.0, 1.0])
        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title('ROC')
        pl.legend(loc="lower right")
        pl.show()
