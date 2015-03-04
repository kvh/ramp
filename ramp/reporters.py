from collections import defaultdict
import copy
import logging
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
try:
    import pylab as pl
except ImportError:
    pl = None
import sklearn

from ramp import metrics
from ramp.utils import pprint_scores


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


class Reporter(object):
    """
    A reporter tracks the results of a series of model runs, as well as summary
    metrics which can be reported in text or graphical form.
    """
    defaults = dict(
            )

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.reset()

    def reset(self):
        self.config = {}
        self.config.update(self.defaults)
        self.config.update(self.kwargs)
        self.summary_df = None
        self.results = []
        self.processed = False

    def copy(self):
        cp = copy.deepcopy(self)
        cp.reset()
        return cp

    def __str__(self):
        if self.summary_df:
            return str(self.summary_df)
        return None

    def _repr_html_(self):
        if self.summary_df is not None:
            return self.summary_df._repr_html_()
        return None

    def build_report(self):
        raise NotImplementedError

    def process_results(self, results):
        self.results = results
        self.build_report()
        self.processed = True

    @staticmethod
    def combine(reporters):
        # TODO: Reporters should know how to combine with other reporters of the same kind to produce aggregate reports.
        raise NotImplementedError

    def plot(self, fig=None, ax=None, **kwargs):
        if ax is None:
            fig, ax = pl.subplots()
        return self._plot(fig, ax, **kwargs)


class ConfusionMatrix(Reporter):

    def build_report(self):
        # TODO: init with labels param
        dfs = []
        vals = pd.concat([result.y_test for result in results]).unique()
        vals.sort()
        cols = ['predicted %s'%c for c in vals]
        rows = ['actual %s'%c for c in vals]
        df = pd.DataFrame(0, index=rows, columns=cols)
        for result in self.results:
            df += pd.DataFrame(sklearn.metrics.confusion_matrix(result.y_test,
                                       result.y_preds), index=rows, columns=cols)
        return df


class RFFeatureImportances(Reporter):

    def build_report(self):
        results = self.results
        importances = pd.DataFrame({'Feature name':[str(f.feature)
                                        for f in results[0].fitted_model.fitted_features]})
        avg_importances = []
        for i in range(len(importances)):
            avg = np.mean([res.fitted_model.fitted_estimator.estimator.feature_importances_[i]
                          for res in results])
            avg_importances.append(avg)
        importances['Average importance'] = avg_importances
        ret = importances.sort('Average importance', ascending=False)
        ret.index = range(len(importances))
        self.summary_df = ret


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
        self.metric = metrics.as_ramp_metric(metric)

    @property
    def name(self):
        return self.metric.name

    def build_report(self):
        lower_quantile = self.config['lower_quantile']
        upper_quantile = self.config['upper_quantile']

        vals = pd.Series([self.metric.score(result) for result in self.results])

        lower_bound = vals.quantile(lower_quantile)
        upper_bound = vals.quantile(upper_quantile)
        median = vals.quantile(0.5)
        mean = vals.mean()

        column_names = [ "Mean" , "Median" , "%d_Percentile" % (lower_quantile*100), "%d_Percentile" % (upper_quantile*100)]
        df = pd.DataFrame(dict(zip(column_names, [mean, median, lower_bound, upper_bound])),
                          columns=column_names,
                          index=[self.metric.name])

        self.metric_values = vals
        self.summary_df = df
        return df

    def _plot(self, fig, ax, index=1, color='b'):
        ax.plot([index]*len(self.results), self.metric_values, '+', mew=2, color=color)
        ax.plot([index], self.summary_df['Mean'], 'rs', alpha=.5)
        ax.set_xticks([index])
        ax.set_xticklabels([self.metric.name])
        return ax


class DualThresholdMetricReporter(MetricReporter):
    """
    Reports on a pair of metrics which are threshold sensitive.

    Thresholds are automatically detected from the evaluation results unless
    explicitly set in the config.
    """

    def __init__(self, metric1, metric2, granularity_sigfigs=3, **kwargs):
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
        if (not isinstance(metric1, metrics.ThresholdMetric)
                or not isinstance(metric2, metrics.ThresholdMetric)):
            raise TypeError("Requires ThresholdMetric")
        self.metric1 = metric1
        self.metric2 = metric2
        self.results = []
        self.n_cached_curves = 0
        self.sigfigs = granularity_sigfigs

    @property
    def n_current_results(self):
        return len(self.results)

    @property
    def thresholds(self):
        if 'thresholds' in self.config:
            return self.config['thresholds']
        else:
            thresholds = pd.concat([result.y_preds for result in self.results])
            thresholds = thresholds.map(lambda x: np.round(x, self.sigfigs)).unique()
            return sorted(list(thresholds))

    def reset_thresholds():
        self.config['thresholds'] = None

    def build_curves(self):
        thresholds = self.thresholds #sorted(list(set(result.y_preds)))
        self.curves = []
        for result in self.results:
            curve = DataFrame(
                    {self.metric1.name: [self.metric1.score(result, threshold) for threshold in thresholds],
                     self.metric2.name: [self.metric2.score(result, threshold) for threshold in thresholds]},
                    index=thresholds)
            self.curves.append(curve)

    def build_report(self):
        """
        Calculates the pair of metrics for each threshold for each result.
        """
        thresholds = self.thresholds
        lower_quantile = self.config['lower_quantile']
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
        self.build_curves()
        self.summary_df = self.ret
        return self.ret

    def plot_error(self, fig_ax=(None, None), color='red', **kwargs):
        curves = self.summary_df

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
        if pl is None:
            logging.error("ERROR: You must install matplotlib in order to see plots")
            return
        fig, ax = fig_ax
        if ax is None:
            fig, ax = pl.subplots()
        for curve in self.curves:
            pl.plot(curve[self.metric1.name],
                    curve[self.metric2.name],
                    # color=color,
                    markeredgecolor=color,
                    **kwargs)
        pl.plot([0, 1], [0, 1], 'k--')
        pl.xlim([0.0, 1.0])
        pl.ylim([0.0, 1.0])
        pl.xlabel(self.metric1.name)
        pl.ylabel(self.metric2.name)
        pl.show()


def combine_dual_reports(reports):
    fig, ax = pl.subplots()
    current_color = 0
    for report in reports:
        current_color = (current_color + 5) % len(colors)
        report.plot(fig_ax=(fig, ax), color=colors[current_color])
    ax.show()

