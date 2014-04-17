from collections import defaultdict

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import pylab as pl
from sklearn import metrics

from ramp.utils import pprint_scores


class Reporter(object):
    defaults = dict(
            verbose=False
            )
    
    def __init__(self, **kwargs):
        self.config = {}
        self.config.update(self.defaults)
        self.config.update(kwargs)
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
        cm = metrics.confusion_matrix(result.y_test, result.y_preds)
        if self.config['verbose']:
            print cm
        self.ret.append(cm)


class MislabelInspector(Reporter):
    def update(self, result):
        for ind in y_test.index:
            a, p = result.y_test.loc[ind], result.y_preds.loc[ind]
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
        p, r, t = metrics.precision_recall_curve(result.y_test, result.y_preds)
        ret = zip(p, r)
        if self.config['verbose']:
            print ret
        self.ret.append()


class ROCCurve(Reporter):
    def optional_kwargs(self):
        return ['verbose']
    
    def update(self, result):
        fpr, tpr, thresholds = metrics.roc_curve(result.y_test, result.y_preds)
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


class MetricReporter(Reporter):
    defaults = dict(
              verbose=False
            , lower_quantile=.05
            , upper_quantile=.95
            )
    
    def __init__(self, metric, **kwargs):
        """
        Accepts a Metric object and evaluates it at each fold.
        """
        Reporter.__init__(self, **kwargs)
        self.metric = metric

    def update(self, result):
        self.ret.append(self.metric.score(result))
    
    def summary_df(self, lower_quantile=None, upper_quantile=None):
        if lower_quantile is None:
            lower_quantile = self.config['lower_quantile']
        if upper_quantile is None:
            upper_quantile = self.config['upper_quantile']
        
        vals = Series(self.ret)
        
        lower_bound = vals.quantile(lower_quantile)
        upper_bound = vals.quantile(upper_quantile)
        median = vals.quantile(50)
        mean = vals.mean()
        
        column_names = [ "Mean" , "Median" , "%d_Percentile" % (lower_quantile*100), "%d_Percentile" % (upper_quantile*100)]
        df = pd.DataFrame(dict(zip(column_names, [mean, median, lower_bound, upper_bound])), index=[0])
        
        return df
    
    def _repr_html_(self):
        return self.summary_df()._repr_html_()
    
    def plot(self):
        vals = Series(self.ret)
        vals.hist()
    
    def report(self, **kwargs):
        """
        Report the results of dual thresholded metrics.
        
        Kwargs:
            Thresholds: list of thresholds. Automatically calculated from the results if not provided.
            lower_quantile: Lower quantile for confidence bound.
            upper_quantile: Upper quantile for confidence bound.
        """
        return self.summary_df(**kwargs)


class DualThresholdMetricReporter(MetricReporter):
    """
    Reports on a pair of metrics which are threshold sensitive.
    
    Thresholds are automatically detected from the evaluation results unless explicitly set in the config.
    """
    
    def __init__(self, metric1, metric2, **kwargs):
        """
        Accepts a Metric object and evaluates it at each fold.
        """
        Reporter.__init__(self, **kwargs)
        self.metric1 = metric1
        self.metric2 = metric2
        self.results = []
        self.n_cached_curves = 0
    
    @property
    def n_current_results(self):
        return len(self.results)
    
    @property
    def thresholds(self):
        if 'thresholds' in self.config:
            return self.config['thresholds']
        else:
            thresholds = set()
            for result in self.results:
                thresholds.update(result.y_preds)
            return list(thresholds)
    
    def update(self, result):
        self.results.append(result)
    
    def summary_df(self, thresholds=None, lower_quantile=None, upper_quantile=None):
        """
        Calculates the pair of metrics for each threshold for each result.
        """
        if thresholds is None:
            thresholds = self.thresholds
        if lower_quantile is None:
            lower_quantile = self.config['lower_quantile']
        if upper_quantile is None:
            upper_quantile = self.config['upper_quantile']
        
        if self.n_current_results > self.n_cached_curves:
            # If there are new curves, recompute
            colnames = ['_'.join([metric, stat])
                        for metric in [self.metric1.name, self.metric2.name] 
                        for stat in ['Mean', 'Median', '%d_Percentile' % (100*lower_quantile), '%d_Percentile' % (upper_quantile*100)]]
            self.ret = pd.DataFrame(columns=colnames, index=thresholds, dtype='float64')
            
            for threshold in thresholds:
                m1s = Series([self.metric1.score(result, threshold) for result in self.results])
                m2s = Series([self.metric2.score(result, threshold) for result in self.results])
                self.ret.loc[threshold] = (m1s.mean(), m1s.quantile(.5), m1s.quantile(.05), m1s.quantile(.95),
                                           m2s.mean(), m2s.quantile(.5), m2s.quantile(.05), m2s.quantile(.95))
        return self.ret
    
    def plot(self, fig_ax=(None, None), color='red', **kwargs):
        curves = self.summary_df()
        
        fig, ax = fig_ax
        if ax is None:
            fig, ax = pl.subplots()

        # Plot medians
        ax.plot(curves[curves.columns[1]].values, 
                curves[curves.columns[5]].values, 
                color=color, markeredgecolor=color)

        # Plot medians
        ax.fill_between(
            curves[curves.columns[1]].values, 
            curves[curves.columns[6]].values, 
            curves[curves.columns[7]].values, 
            facecolor=color, edgecolor='', interpolate=True, alpha=.33)
        
        if fig is None:
            return ax
        else:
            return fig, ax


def combine_dual_reports(reports):
    colors = [ '#723C95'
             , '#5697D5'
             , '#9BCECA'
             , '#92BC45'
             , '#C7D632'
             , '#F6E400'
             , '#ECB61B'
             , '#E08C2C'
             , '#D3541F'
             , '#CD1E20'
             , '#C64A98'
             , '#A34F9B']
    
    fig, ax = pl.subplots()
    current_color = 0
    for report in reports:
        current_color = (current_color + 5) % len(colors)
        reports.plot(ax=ax, color=colors[current_color])
    ax.show()


