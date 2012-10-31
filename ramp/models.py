from utils import make_folds, _pprint
from pandas import Series, concat
import random
import hashlib
import copy
import numpy as np
from sklearn import cross_validation, ensemble, linear_model
from builders import build_featureset, build_target

debug = False

""" model fitting has no caching currently. for one, not that useful
since you usually want to run different models/features/training subsets. also,
hard to implement (so many variables/data to key on, are they all really immutable?)
"""

def get_xy(config, context):
    x = build_featureset(config.features, context)
    y = build_target(config.target, context)

    if config.column_subset:
        x = x[config.column_subset]
    return x, y


def get_key(config, context):
    return '%r-%s' % (config, context.create_key())


def fit(config, context):
    x, y = None, None
    try:
        # model caching
        config.model = context.store.load(get_key(config, context))
        print "loading stored model..."
    except KeyError:
        x, y = get_xy(config, context)

        train_x = x.reindex(context.train_index)
        train_y = y.reindex(context.train_index)

        if debug:
            print train_x
        if debug:
            print "Fitting model '%s'." % (config.model.__name__)

        config.model.fit(train_x.values, train_y.values)
        context.store.save(get_key(config, context), config.model)

    config.update_reporters_with_model(config.model)

    return x, y


def predict(config, context, predict_index, fit_model=True):
    if (context.train_index & predict_index):
        print "WARNING: train and predict indices overlap..."

    if fit_model:
        x, y = fit(config, context)

    # TODO: possible to have x loaded without new prediction rows
    if x is None:
        # rebuild just the necessary x:
        ctx = context.copy()
        ctx.data = context.data.ix[predict_index]
        x, y = get_xy(config, ctx)

    # ensure correct columns exist:
#    for col in columns_used:
#        if col not in x.columns:
#            print "WARNING: filling missing column '%s' with zeros" % col
#            x[col] = Series(np.random.randn(len(x)) / 100, index=x.index)
#    symdif = set(x.columns) ^ set(columns_used)
#    if symdif:
#        print symdif
#        raise Exception("mismatched columns between fit and predict.")
    # re-order columns
#    x = x.reindex(columns=columns_used)

    predict_x = x.reindex(predict_index)

    # make actual predictions
    ps = config.model.predict(predict_x.values)
    preds = Series(ps, index=predict_x.index)

    # prediction post-processing
    if config.prediction is not None:
        context.data[config.predictions_name] = preds
        preds = build_target(config.prediction, context)
        preds = get_single_column(preds).reindex(predict_x.index)
    preds.name = ''
    return preds, x, y


def cv(config, context, folds=5, repeat=2, save=False):
    # TODO: too much overloading on folds here
    if isinstance(folds, int):
        folds = make_folds(context.data.index, folds, repeat)
    scores = []
    for train, test in folds:
        context.train_index = train
        preds, x, y = predict(config, context, test)
        actuals = y.reindex(test)
        config.update_reporters_with_predictions(context, x, actuals, preds)
        scores.append(config.metric.score(actuals,
            preds))
    scores = np.array(scores)
    #if save:
        #dataset.save_models([(scores, copy.copy(config))])
    return scores


def print_scores(scores):
    print "%0.3f (+/- %0.3f) [%0.3f,%0.3f]" % (
          scores.mean(), scores.std(), min(scores),
          max(scores))


def build_model(config, context, name=None):
    models.fit(config, context)
    context.store.save('model__%s' % name, get_key(config, context))


def get_or_build_model(config, context, name):
    try:
        key = context.store.load('model__%s' % name)
        config.model = context.store.load(key)
    except KeyError:
        build_model(config, context, name)

