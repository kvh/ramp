import pandas as pd
try:
    import pylab as pl
except ImportError:
    pl = None

from ramp.model_definition import ModelDefinition, model_definition_factory
from ramp import metrics, modeling, reporters
from ramp.reporters import MetricReporter, colors


class CVResult(object):

    def __init__(self, results, reporters=None, metrics=None):
        self.results = results
        self.reporters = reporters
        self.metrics = metrics
        self.model_def = self.results[0].model_def
        for r in metrics + reporters:
            if not r.processed:
                r.process_results(self.results)

    def __repr__(self):
        return repr(self.summary_df())

    def _repr_html_(self):
        return self.summary_df()._repr_html_()

    def summary_df(self):
        df = pd.concat([r.summary_df for r in self.metrics])
        df.index = [m.name for m in self.metrics]
        return df

    def summary(self):
        return self.summary_df()

    def plot(self):
        fig, axes = pl.subplots(1, len(self.metrics))
        for i, m in enumerate(self.metrics):
            m.plot(fig=fig, ax=axes[i])

    def classification_curve(self, x_metric, y_metric):
        x_metric = metrics.as_ramp_metric(x_metric)
        y_metric = metrics.as_ramp_metric(y_metric)
        dtmr = reporters.DualThresholdMetricReporter(x_metric,
                                                     y_metric)
        dtmr.process_results(self.results)
        return dtmr

    def feature_importances(self):
        reporter = reporters.RFFeatureImportances()
        return self.build_report(reporter)

    def build_report(self, report):
        report.process_results(self.results)
        return report


class CVComparisonResult(object):

    def __init__(self, cvresults_dict):
        self.cvresults = cvresults_dict.values()
        self.model_defs = cvresults_dict.keys()
        self.metrics = self.cvresults[0].metrics
        self.reporters = self.cvresults[0].reporters
        self.model_abbrs = {"Model %d" % (i+1): md for i, md in enumerate(self.model_defs)}
        self.n = len(self.cvresults)

    def __repr__(self):
        return '\n'.join("%s\n%s" % (k, repr(v.summary_df())))

    def _repr_html_(self):
        return self.summary_df()._repr_html_()

    def summary_df(self):
        df = pd.concat([r.summary_df() for r in self.cvresults])

        df.index = pd.MultiIndex.from_product([self.model_abbrs.keys(),
                                               [m.name for m in self.metrics]])
        return df

    def summary(self):
        return self.summary_df()

    def model_legend(self):
        df = pd.DataFrame([cvr.model_def.describe() for cvr in self.cvresults])
        df.index = self.model_abbrs.keys()
        return df

    def plot(self):
        fig, axes = pl.subplots(1, len(self.metrics))
        fig.set_size_inches(12, 6)
        fig.tight_layout()
        for i, result in enumerate(self.cvresults):
            for m, ax in zip(result.metrics, axes):
                clr = colors[i % len(colors)]
                m.plot(fig, ax, index=i, color=clr)
        for m, ax in zip(self.metrics, axes):
            ax.set_xlim(-0.5, self.n - 0.5)
            ax.set_xticks(range(self.n))
            ax.set_title(m.metric.name)
            ax.set_xticklabels(self.model_abbrs.keys(),
                               rotation=45 + min(1, self.n / 10) * 35)
            ax.autoscale(True, 'y')


def cross_validate(data=None, folds=5, repeat=1, metrics=None,
                   reporters=None, model_def=None, **kwargs):
    """Shortcut to cross-validate a single configuration.

    ModelDefinition variables are passed in as keyword args, along
    with the cross-validation parameters.
    """
    md_kwargs = {}
    if model_def is None:
        for arg in ModelDefinition.params:
            if arg in kwargs:
                md_kwargs[arg] = kwargs.pop(arg)
        model_def = ModelDefinition(**md_kwargs)
    if metrics is None:
        metrics = []
    if reporters is None:
        reporters = []
    metrics = [MetricReporter(metric) for metric in metrics]
    results = modeling.cross_validate(model_def, data, folds, repeat=repeat, **kwargs)
    for r in reporters + metrics:
        r.process_results(results)
    return CVResult(results, reporters, metrics)


def cv_factory(data=None, folds=5, repeat=1, reporters=[], metrics=None, **kwargs):
    """Shortcut to iterate and cross-validate models.

    All ModelDefinition kwargs should be iterables that can be
    passed to model_definition_factory.
    """
    cv_runner = kwargs.pop('cv_runner', cross_validate)
    md_kwargs = {}
    for arg in ModelDefinition.params:
        if arg in kwargs:
            md_kwargs[arg] = kwargs.pop(arg)
    model_def_fact = model_definition_factory(ModelDefinition(), **md_kwargs)
    results = {}
    for model_def in model_def_fact:
        reporters = [reporter.copy() for reporter in reporters]
        cvr = cv_runner(model_def=model_def,
                        data=data,
                        folds=folds,
                        repeat=repeat,
                        reporters=reporters,
                        metrics=metrics,
                        **kwargs)
        results[model_def] = cvr

    return CVComparisonResult(results)
