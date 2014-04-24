from collections import defaultdict
import logging
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import pylab as pl
from sklearn import metrics

from ramp.utils import pprint_scores

class Reporter(object):
    """
    A reporter tracks the results of a series of model runs, as well as summary
    metrics which can be reported in text or graphical form.
    """
    defaults = dict(
            )
    summary = []
    results = []
    
    @classmethod
    def factory(cls, *args, **kwargs):
        """
        Provides a function generating reporter objects with a given set of
        initialization parameters.
        """
        return lambda: cls(*args, **kwargs)
    
    def __init__(self, **kwargs):
        self.config = {}
        self.config.update(self.defaults)
        self.config.update(kwargs)
    
    def __str__(self):
        return str(self.summary)
    
    def recompute(self):
        self.summary = [self.summarize(result) for result in self.results]
    
    def summarize(self, result):
        raise NotImplementedError
    
    def update(self, result):
        """
        Accepts an object of type Result and updates the report's internal representation.
        """
        self.results.append(result)
        ret = self.summarize(result)
        self.summary.append(ret)
        logging.debug("{name}.update returned {ret}".format(name=self.__class__.__name__, ret=ret))
    
    @staticmethod
    def combine(reporters):
        # TODO: Reporters should know how to combine with other reporters of the same kind to produce aggregate reports.
        raise NotImplementedError

class ModelOutliers(Reporter):
    pass

class ConfusionMatrix(Reporter):
    @staticmethod
    def summarize(result):
        return metrics.confusion_matrix(result.y_test, result.y_preds)

class MislabelInspector(Reporter):
    @staticmethod
    def summarize(result):
        ret = []
        for ind in result.y_test.index:
            a, p = result.y_test.loc[ind], result.y_preds.loc[ind]
            if a != p:
                ret_strings = ["-" * 20]
                ret_strings.append("Actual: %s\tPredicted: %s" % (a, p))
                ret_strings.append(result.x_test.ix[ind])
                ret_strings.append(result.y_test.ix[ind])
                ret.append(ret_strings)
        return '\n'.join(ret)

class RFImportance(Reporter):
    importance_arrays = []
    
    @property
    def summary(self):
        """
        Returns feature importances, sorted with most important features first.
        """
        return pd.concat(self.importance_arrays, axis=1).mean(axis=1).sort(ascending=False)
    
    def update(self, result):
        try:
            imps = result.fitted_model.feature_importances_
        except AttributeError:
            logging.warning("Warning: Model has no feature importances")
            return
        if imps is None:
            logging.warning("Warning: Model has no feature importances")
            return
        self.importance_arrays.append(imps)
        logging.debug(imps) 
        self.results.append(result)
    
    def _repr_html_(self):
        return self.summary._repr_html_()
    
class PRCurve(Reporter):
    def summarize(self, result):
        p, r, t = metrics.precision_recall_curve(result.y_test, result.y_preds)
        ret = DataFrame(
                {'Precision': p, 
                 'Recall': r}, 
                index=t)
        return ret
    
    def plot(self):
        try:
            import pylab as pl
        except ImportError as e:
            logging.error("ERROR: You must install matplotlib in order to see plots")
            return
        for curve in self.summary:
            pl.plot(curve['Precision'], curve['Recall'])
        pl.plot([0, 1], [0, 1], 'k--')
        pl.xlim([0.0, 1.0])
        pl.ylim([0.0, 1.0])
        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title('ROC')
        pl.legend(loc="lower right")
        pl.show()

class ROCCurve(Reporter):
    def summarize(self, result):
        fpr, tpr, thresholds = metrics.roc_curve(result.y_test, result.y_preds)
        ret = DataFrame(
                {'FPR':fpr, 
                 'TPR': tpr},
                index=thresholds)
        return ret
    
    def plot(self):
        try:
            import pylab as pl
        except ImportError as e:
            logging.error("ERROR: You must install matplotlib in order to see plots")
            return
        for curve in self.summary:
            pl.plot(curve['FPR'], curve['TPR'])
        pl.plot([0, 1], [0, 1], 'k--')
        pl.xlim([0.0, 1.0])
        pl.ylim([0.0, 1.0])
        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title('ROC')
        pl.legend(loc="lower right")
        pl.show()

class OOBEst(Reporter):
    def update(self, model):
        try:
            ret = self.summary.append(model.oob_score_)
            logging.debug("OOB score:" + model.oob_score_)
        except AttributeError:
            logging.exception("Model has no OOB score")
    
    def __str__(self):
        if not self.summary:
            return None
        else:
            return "OOB Est: %s" % (pprint_scores(self.summary))

class MetricReporter(Reporter):
    defaults = dict(
              lower_quantile=.05
            , upper_quantile=.95
            )
    
    def __init__(self, metric, **kwargs):
        """
        Accepts a Metric object and evaluates it at each fold.
        """
        Reporter.__init__(self, **kwargs)
        self.metric = metric
        self.summarize = self.metric.score
    
    def summary_df(self):
        lower_quantile = self.config['lower_quantile']
        upper_quantile = self.config['upper_quantile']
        
        vals = Series(self.summary)
        
        lower_bound = vals.quantile(lower_quantile)
        upper_bound = vals.quantile(upper_quantile)
        median = vals.quantile(0.5)
        mean = vals.mean()
        
        column_names = [ "Mean" , "Median" , "%d_Percentile" % (lower_quantile*100), "%d_Percentile" % (upper_quantile*100)]
        df = pd.DataFrame(dict(zip(column_names, [mean, median, lower_bound, upper_bound])), index=[0])
        
        return df
    
    def __str__(self):
        return str(self.summary_df())
    
    def _repr_html_(self):
        return self.summary_df()._repr_html_()
    
    def plot(self):
        vals = Series(self.summary)
        ax = vals.hist()
        ax.set_title("%s Histogram" % self.metric.name)
        return ax


class DualThresholdMetricReporter(MetricReporter):
    """
    Reports on a pair of metrics which are threshold sensitive.
    
    Thresholds are automatically detected from the evaluation results unless
    explicitly set in the config.
    """
    
    def __init__(self, metric1, metric2, **kwargs):
        """
        Accepts a Metric object and evaluates it at each fold.
        
        Parameters:
        Thresholds: [Float], default: None
            Thresholds for reporting the two metrics. If not set, thresholds
            are model predictions from all runs computed on the fly.
        
        lower_quantile, upper_quantile: Float, default 0.05, 0.095
            Quantiles to be used for reporting metrics.
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
            thresholds = pd.Series()
            for result in self.results:
                print result.y_preds
                thresholds = pd.concat(thresholds, result.y_preds).unique()
                print thresholds
            return list(thresholds)
    
    def reset_thresholds():
        self.config['thresholds'] = None
    
    def summarize(self, result):
        thresholds = sorted(list(set(result.y_preds)))
        ret = DataFrame(
                {self.metric1.name: [self.metric1.score(result, threshold) for threshold in thresholds],
                 self.metric2.name: [self.metric2.score(result, threshold) for threshold in thresholds]},
                index=thresholds)
        logging.debug("preds:")
        logging.debug(result.y_preds)
        logging.debug("thresholds:")
        logging.debug(thresholds)
        return ret
    
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
                        for stat in ['Mean', 'Median',
                                     '%d_Percentile' % (100*lower_quantile),
                                     '%d_Percentile' % (upper_quantile*100)]]
            self.ret = pd.DataFrame(columns=colnames, index=thresholds, dtype='float64')
            
            for threshold in thresholds:
                m1s = Series([self.metric1.score(result, threshold) for result in self.results])
                m2s = Series([self.metric2.score(result, threshold) for result in self.results])
                self.ret.loc[threshold] = (m1s.mean(), m1s.quantile(.5), m1s.quantile(.05), m1s.quantile(.95),
                                           m2s.mean(), m2s.quantile(.5), m2s.quantile(.05), m2s.quantile(.95))
        return self.ret
    
    def plot_error(self, fig_ax=(None, None), color='red', **kwargs):
        curves = self.summary_df()
        
        fig, ax = fig_ax
        if ax is None:
            fig, ax = pl.subplots()
        
        # Plot medians
        ax.plot(curves[curves.columns[1]].values, 
                curves[curves.columns[5]].values, 
                color=color, 
                markeredgecolor=color)
        
        # Plot medians
        ax.fill_between(
            curves[curves.columns[1]].values, 
            curves[curves.columns[6]].values, 
            curves[curves.columns[7]].values, 
            facecolor=color, edgecolor='', interpolate=True, alpha=.33)
        ax.set_xlabel(self.metric1.name.capitalize())
        ax.set_ylabel(self.metric2.name.capitalize())
        
        if fig is None:
            return ax
        else:
            return fig, ax
    
    def plot(self, fig_ax=(None, None), color='red', **kwargs):
        fig, ax = fig_ax
        if ax is None:
            fig, ax = pl.subplots()
        for curve in self.summary:
            pl.plot(curve[self.metric1.name], 
                    curve[self.metric2.name],
                    color=color, 
                    markeredgecolor=color,
                    **kwargs)
        pl.plot([0, 1], [0, 1], 'k--')
        pl.xlim([0.0, 1.0])
        pl.ylim([0.0, 1.0])
        pl.xlabel(self.metric1.name)
        pl.ylabel(self.metric2.name)
        pl.show()


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
        reports.plot(fig_ax=(fig, ax), color=colors[current_color])
    ax.show()


