from utils import make_folds, _pprint, get_single_column, pprint_scores
from pandas import Series, concat, DataFrame
import random
import hashlib
import copy
import numpy as np
from sklearn import cross_validation, ensemble, linear_model
from builders import build_featureset, build_target
from prettytable import PrettyTable, ALL

debug = False

""" 
Model fitting has no caching currently. For one, not that
useful since you usually want to run different
models/features/training subsets. Also, hard to implement (so
many variables/data to key on, are they all really immutable?)
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


def get_metric_name(metric):
    if hasattr(metric, 'name'):
        name = metric.name
    else:
        name = metric.__name__
    return name


def fit(config, context, model_name=None, load_only=False):
    x, y = None, None
    try:
        # model caching
        config.model = context.store.load(model_name or get_key(config, context))
        print "Loading stored model..."
    except KeyError:
        if load_only:
            raise Exception("Could not load model and load_only=True.")
        x, y = get_xy(config, context)

        train_x = x.reindex(context.train_index)
        train_y = y.reindex(context.train_index)

        config.model.column_names = train_x.columns

        if debug:
            print train_x
            print train_x.columns

        print "Fitting model '%s' ... " % (config.model),
        #if isinstance(config.model, DataFrameEstimator):
            #config.model.fit(train_x, train_y)
        #else:
        config.model.fit(train_x.values, train_y.values)
        print "[OK]"
        context.store.save(model_name or get_key(config, context), config.model)

    config.update_reporters_with_model(config.model)

    return x, y


def predict(config, context, predict_index, fit_model=True, model_name=None):
    if len(context.train_index & predict_index):
        print "WARNING: train and predict indices overlap..."

    x, y = None, None

    if model_name:
        config.model = context.store.load(model_name)

    if not model_name and fit_model:
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

    predict_x = x.reindex(predict_index)

    print "Making predictions... ",
    # make actual predictions
    ps = config.model.predict(predict_x.values)
    try:
        preds = Series(ps, index=predict_x.index)
    except:
        preds = DataFrame(ps, index=predict_x.index)
    print "[OK]"
    # prediction post-processing
    if config.prediction is not None:
        old = context.data
        context.data = context.data.reindex(predict_x.index)
        context.data[config.predictions_name] = preds
        preds = build_target(config.prediction, context)
        preds = preds.reindex(predict_x.index)
        context.data = old
    preds.name = ''
    actuals = y.reindex(predict_index)
    # TODO: handle multi-variate predictions
    predict_x['predictions'] = preds
    predict_x['actuals'] = actuals
    config.update_reporters_with_predictions(context, predict_x, actuals, preds)
    return predict_x


def cv(config, context, folds=5, repeat=2, print_results=False,
       predict_method=None, predict_update_column=None):
    # TODO: too much overloading on folds here
    if isinstance(folds, int):
        folds = make_folds(context.data.index, folds, repeat)
    #else:
        #folds.set_context(config, context)
    scores = {get_metric_name(m): [] for m in config.metrics}
    # we are overwriting indices, so make a copy
    ctx = context.copy()
    i = 0
    folds = list(folds)
    k = len(folds)/repeat
    for train, test in folds:
        print "\nCross-Validation fold %d/%d round %d/%d" % (i % k + 1, k, i/k + 1, repeat)
        i += 1
        ctx.train_index = train
        ctx.test_index = test
        fold_scores, result = evaluate(config, ctx, test, predict_method, predict_update_column)
        context.latest_result = result
        for metric_, s in fold_scores.items():
            scores[metric_].append(s)
        if print_results:
            for metric_, s in scores.items():
                print "%s: %s" % (metric_, pprint_scores(s))
    result = {'config':config, 'scores':scores}

    # report results
    t = PrettyTable(["Reporter", "Report"])
    t.hrules = ALL
    t.align["Reporter"] = "l"
    t.align["Report"] = "l"
    for reporter in config.reporters:
        t.add_row([reporter.__class__.__name__, reporter.report()])
        reporter.reset()
    print t
    
    return result


def evaluate(config, ctx, predict_index,
             predict_method=None, predict_update_column=None):
    if predict_method is None:
        result = predict(config, ctx, predict_index)
    else:
        # TODO: hacky!
        result = predict_method(config, ctx, predict_index, update_column=predict_update_column)
    preds = result['predictions']
    y = result['actuals']

    try:
        if config.actual is not None:
            actuals = build_target(config.actual, ctx).reindex(predict_index)
        else:
            actuals = y.reindex(predict_index)
    #TODO: HACK -- there may not be an actual attribute on the config
    except AttributeError:
        actuals = y.reindex(predict_index)

    scores = {}
    for metric in config.metrics:
        name = get_metric_name(metric)
        if hasattr(metric, 'score'):
            scores[name] = metric.score(actuals, preds)
        else:
            scores[name] = metric(actuals, preds)
    return scores, result


def predict_autosequence(config, context, predict_index, fit_model=True, update_column=None):
    if len(context.train_index & predict_index):
        print "WARNING: train and predict indices overlap..."

    x, y = None, None

    if fit_model:
        x, y = fit(config, context)

    if debug:
        print x.columns
        print config.model.coef_

    ctx = context.copy()
    ps = []
    for i in predict_index:
        ctx.data = context.data
        x = get_x(config, ctx)
        predict_x = x.reindex([i])

        # make actual predictions
        p = config.model.predict(predict_x.values)
        if update_column is not None:
            ctx.data[update_column][i] = p[0]
        ps.append(p[0])
    try:
        preds = Series(ps, index=predict_index)
    except:
        preds = DataFrame(ps, index=predict_index)
    # prediction post-processing
    if config.prediction is not None:
        context.data[config.predictions_name] = preds
        preds = build_target(config.prediction, context)
        preds = preds.reindex(predict_index)
    preds.name = ''
    return preds, x, y


def cv_autosequence(config, context, folds=5, repeat=2, print_results=False, update_column=None):
    return cv(config, context, folds, repeat, print_results, predict_method=predict_autosequence,
            predict_update_column=update_column)

