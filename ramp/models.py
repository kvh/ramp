from utils import make_folds, _pprint
from pandas import Series, concat, DataFrame
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

def get_x(config, context):
    x = build_featureset(config.features, context)
    if config.column_subset:
        x = x[config.column_subset]
    return x

def get_y(config, context):
    return build_target(config.target, context)

def get_xy(config, context):
    return get_x(config, context), get_y(config, context)


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
            print "Fitting model '%s'." % (config.model)

        config.model.fit(train_x.values, train_y.values)
        context.store.save(get_key(config, context), config.model)

    config.update_reporters_with_model(config.model)

    return x, y


def predict(config, context, predict_index, fit_model=True):
    if len(context.train_index & predict_index):
        print "WARNING: train and predict indices overlap..."

    x, y = None, None

    if fit_model:
        x, y = fit(config, context)

    # TODO: possible to have x loaded without new prediction rows
    if x is None:
        # rebuild just the necessary x:
        ctx = context.copy()
        ctx.data = context.data.ix[predict_index]
        x = get_x(config, ctx)
        try:
            # we may or may not have y's in predict context
            # we get them if we can for metrics and reporting
            y = get_y(config, ctx)
        except KeyError:
            pass

    if debug:
        print x.columns
        print config.model.coef_

    predict_x = x.reindex(predict_index)

    # make actual predictions
    ps = config.model.predict(predict_x.values)
    try:
        preds = Series(ps, index=predict_x.index)
    except:
        preds = DataFrame(ps, index=predict_x.index)

    # prediction post-processing
    if config.prediction is not None:
        context.data[config.predictions_name] = preds
        preds = build_target(config.prediction, context)
        preds = get_single_column(preds).reindex(predict_x.index)
    preds.name = ''
    return preds, x, y


def cv(config, context, folds=5, repeat=2, print_results=False):
    # TODO: too much overloading on folds here
    if isinstance(folds, int):
        folds = make_folds(context.data.index, folds, repeat)
    scores = dict([(m.name, []) for m in config.metrics])
    # we are overwriting indices, so make a copy
    ctx = context.copy()
    for train, test in folds:
        ctx.train_index = train
        preds, x, y = predict(config, ctx, test)
        actuals = y.reindex(test)
        config.update_reporters_with_predictions(ctx, x, actuals, preds)
        for metric in config.metrics:
            scores[metric.name].append(
                    metric.score(actuals,preds))
    #if save:
        #dataset.save_models([(scores, copy.copy(config))])
    if print_results:
        print "\n" + str(config)
        print_scores(scores)
    return scores


def print_scores(scores_dict):
    for metric, scores in scores_dict.items():
        scores = np.array(scores)
        print metric
        print "%0.4f (+/- %0.4f) [%0.4f,%0.4f]\n" % (
            scores.mean(), scores.std(), min(scores),
            max(scores))

