import models
from configuration import Configuration
from context import DataContext


def fit(store=None, data=None, **kwargs):
    return models.fit(Configuration(**kwargs), DataContext(store, data))


def predict(store=None, data=None, predict_index=None, **kwargs):
    if predict_index is None:
        raise ValueError("You must specify a predict_index kw arg")
    return models.predict(Configuration(**kwargs),
            DataContext(store, data), predict_index=predict_index)


def cv(store=None, data=None, **kwargs):
    fargs = {}
    cvargs = ['folds', 'repeat', 'print_results']
    for arg in cvargs:
        if arg in kwargs:
            fargs[arg] = kwargs.pop(arg)
    return models.cv(Configuration(**kwargs),
            DataContext(store, data), **fargs)
