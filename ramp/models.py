from utils import make_folds, _pprint
from pandas import Series, concat
import random
import hashlib
import copy
import numpy as np
from sklearn import cross_validation, ensemble, linear_model

debug = False


def fit(dataset, config, index=None):
    x = dataset.get_train_x(config.features, index)
    y = dataset.get_train_y(config.target, index)
    # print "predict x, y", x, y
    # print "columns", config.column_subset
    if config.column_subset:
        x = x[config.column_subset]
    if debug:
        print x
    if debug:
        print "Fitting model '%s' for dataset '%s'." % (config.model.__name__,
                dataset.display_name)
    config.model.fit(x.values, y.values)
    return x.columns

def predict(dataset, config, index, train_index=None, train_dataset=None):
    if train_dataset is None:
        train_dataset = dataset
    columns_used = fit(train_dataset, config, train_index)
    x = dataset.get_x(config.features, train_index).reindex(index)
    if config.column_subset:
        x = x[config.column_subset]
    if debug:
        print x
    # ensure correct columns exist:
    for col in columns_used:
        if col not in x.columns:
            print "WARNING: filling missing column '%s' with zeros" % col
            x[col] = Series(np.random.randn(len(x)) / 100, index=x.index)
    symdif = set(x.columns) ^ set(columns_used)
    if symdif:
        print symdif
        raise Exception("mismatched columns between fit and predict.")
    # re-order columns
    x = x.reindex(columns=columns_used)
    ps = config.model.predict(x.values)
    preds = Series(ps, index=x.index)
    if config.prediction is not None:
        dataset.data[config.predictions_name] = preds
        preds = dataset.make_feature(config.prediction, train_index, force=True)
        preds = preds[preds.columns[0]].reindex(x.index)
    preds.name = ''
    return preds

def cv(dataset, config, folds=5, repeat=1, save=False):
    idx = dataset.train_index
    # TODO: too much overloading on folds here
    if isinstance(folds, int):
        folds = make_folds(idx, folds, repeat)
    scores = []
    for train, test in folds:
        preds = predict(dataset, config, test, train)
        scores.append(config.metric.score(dataset.get_train_y(config.actual, test),
            preds))
    scores = np.array(scores)
    if save:
        dataset.save_models([(scores, copy.copy(config))])
    return scores

def print_scores(scores):
    print "%0.3f (+/- %0.3f) [%0.3f,%0.3f]" % (
          scores.mean(), scores.std(), min(scores),
          max(scores))



