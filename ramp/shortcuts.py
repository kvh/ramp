import models
from configuration import Configuration, ConfigFactory
from context import DataContext


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
    for conf in fact:
        models.cv(conf, DataContext(store, data), **fargs)
