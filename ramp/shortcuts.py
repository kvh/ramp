import models
from configuration import Configuration, ConfigFactory
from context import DataContext
from prettytable import PrettyTable, ALL
import numpy as np


def fit(store=None, data=None, **kwargs):
    return models.fit(Configuration(**kwargs), DataContext(store, data))


def predict(store=None, data=None, predict_index=None, **kwargs):
    if predict_index is None:
        raise ValueError("You must specify a predict_index kw arg")
    return models.predict(Configuration(**kwargs),
            DataContext(store, data), predict_index=predict_index)


def cv(store=None, data=None, **kwargs):
    """Shortcut to cross-validate a single configuration.

    Config variables are passed in as keyword args, along
    with the cross-validation parameters.
    """
    fargs = {}
    cvargs = ['folds', 'repeat', 'print_results']
    for arg in cvargs:
        if arg in kwargs:
            fargs[arg] = kwargs.pop(arg)
    return models.cv(Configuration(**kwargs),
            DataContext(store, data), **fargs)


def cv_factory(store=None, data=None, **kwargs):
    """Shortcut to iterate and cross-validate configurations.

    All configuration kwargs should be iterables that can be
    passed to a ConfigFactory.
    """
    fargs = {'print_results':True}
    cvargs = ['folds', 'repeat', 'print_results']
    for arg in cvargs:
        if arg in kwargs:
            fargs[arg] = kwargs.pop(arg)
    fact = ConfigFactory(Configuration(), **kwargs)
    results = []
    for conf in fact:
        results.append(models.cv(conf, DataContext(store, data), **fargs))
    t = PrettyTable(["Configuration", "Score"])
    t.hrules = ALL
    t.align["Config"] = "l"
    for r in results:
        scores_dict = r['scores']
        s = ""
        for metric, scores in scores_dict.items():
            scores = np.array(scores)
            s += "%s: %0.4f (+/- %0.4f) [%0.4f,%0.4f]\n" % (
                    metric,
                    scores.mean(), scores.std(), min(scores),
                    max(scores))
        t.add_row([str(r['config']), s])
    print t
