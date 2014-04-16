from sklearn import metrics
import numpy as np
from pandas import DataFrame
from utils import pprint_scores
from collections import defaultdict


class Reporter(object):
    def __init__(self, **kwargs):
        self.config = {}
        for kwarg in self.optional_kwargs():
            self.config[kwarg] = None
        self.config.extend(kwargs)
        self.ret = []

    def optional_kwargs(self):
        """
        Used to avoid attribute errors for optional keyword arguments, this
        function returns a list of possible keyword arguments.
        """
        return ['verbose']
    
    def update(self, result):
        """
        Accepts an object of type Result and updates the report's internal representation.
        """
        pass
    
    def report(self):
        """
        Output the report. 
        A report's output can be an image, text string, or even a web page.
        Due to the variability in possible outputs, reports are often returned
        via side effects.
        """
        return self.ret

class ModelOutliers(Reporter):
    pass

class ConfusionMatrix(Reporter):
    def update(self, result):
        # TODO: Make sure result.evals makes sense here, might need to subset.
        cm = metrics.confusion_matrix(result.y_test, result.evals)
        try:
            factors = result.fitted_model.prep_data
        except KeyError:
            print cm
            return
        if factors:
            names = [f[0] for f in factors]
            df = DataFrame(cm, columns=names, index=names)
            if config.verbose:
                print df.to_string()
            self.ret.append(df.to_string())
        else:
            if config.verbose:
                print cm
            self.ret.append(cm)

class MislabelInspector(Reporter):
    def update(self, result):
        for ind in y_test.index:
            a, p = y_test[ind], evals[ind]
            if a != p:
                ret_strings = ["-" * 20]
                ret_strings.append("Actual: %s\tPredicted: %s" % (a, p))
                ret_strings.append(result.x_test.ix[ind])
                ret_strings.append(result.y_test.ix[ind])
                ret = ''.join(ret_strings, '\n')
                if self.config['verbose']:
                    print ret
                self.ret.append(ret)

class RFImportance(Reporter):
    def update(self, result):
        try:
            imps = result.fitted_model.feature_importances_
        except AttributeError:
            print "Warning: Model has no feature importances"
            return
        if imps is None:
            print "Warning: Model has no feature importances"
            return
        imps = sorted(zip(imps, result.fitted_model.column_names), reverse=True)
        if self.config['verbose']:
            print self.print_string(imps)
        self.ret.append(imps)
    
    def print_string(self, imps):
        s = "Average Feature Importances\n"
        s += "Rank\tGini\tFeature\n"
        for i, x in enumerate(imps):
            imp, f = x
            s += '%d\t%0.4f\t%s\n'%(i+1,np.average(imp), f)
        return s
    
    def report(self):
        if not self.ret:
            return
        d = defaultdict(list)
        for imps in self.ret:
            for imp, f in imps:
                d[f].append(imp)
        ret = sorted(zip(d.values(), d.keys()), key=lambda x: np.average(x[0]), reverse=True)
        if self.config['verbose']:
            self.print_string(ret)
        return ret

class PRCurve(Reporter):
    def update(self, result):
        p, r, t = metrics.precision_recall_curve(result.y_test, result.evals)
        ret = zip(p, r)
        if self.config['verbose']:
            print ret
        self.ret.append()

class ROCCurve(Reporter):
    def optional_kwargs(self):
        return ['verbose']
    
    def update(self, result):
        fpr, tpr, thresholds = metrics.roc_curve(result.y_test, result.evals)
        self.ret.append((fpr, tpr, thresholds))
        if self.config['verbose']:
            print "ROC thresholds"
            print "FP Rate\tTP Rate"
            for x in zip(fpr, tpr):
                print '%0.4f\t%0.4f' % x
    
    def report(self):
        try:
            import pylab as pl
        except ImportError as e:
            print "ERROR: You must install matplotlib in order to see plots"
            return
        for fpr, tpr, thresholds in self.ret:
            pl.plot(fpr, tpr, label='ROC curve')
        pl.plot([0, 1], [0, 1], 'k--')
        pl.xlim([0.0, 1.0])
        pl.ylim([0.0, 1.0])
        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title('ROC')
        pl.legend(loc="lower right")
        pl.show()

class OOBEst(Reporter):
    def __init__(self):
        self.scores = []
    
    def update_with_model(self, model):
        try:
            if self.config['verbose']:
                print "OOB score:", model.oob_score_
            self.ret.append(model.oob_score_)
        except AttributeError:
            print "Model has no OOB score"
    
    def report(self):
        if not self.ret:
            return
        return "OOB Est: %s" % (pprint_scores(self.ret))

