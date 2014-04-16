from pandas import Series, DataFrame, concat

from ramp.builders import build_target_safe
from ramp.features.base import to_feature, ComboFeature, Feature, DummyFeature
from ramp.modeling import fit_model, predict_model
from ramp.utils import make_folds, get_single_column


class Predictions(Feature):
    # TODO: update for new context

    def __init__(self, model_def, name=None, external_context=None,
            cv_folds=None, cache=False):
        """
        If cv-folds is specified, will use k-fold cross-validation to
        provide robust predictions.
        (The predictions returned are those predicted on hold-out sets only.)
        Will not provide overly-optimistic fit like Predictions will, but can
        increase runtime significantly (nested cross-validation).
        Can be int or iteratable of (train, test) indices
        """
        self.cv_folds = cv_folds
        self.model_def = model_def
        self.external_context = external_context
        self.feature = DummyFeature()
        self._cacheable = cache
        self.trained = True
        super(Predictions, self).__init__(self.feature)
        # if self.external_context is not none:
        #     # dont need to retrain if using external dataset to train
        #     self.trained = false
        if not name:
            name = 'predictions'
        self._name = '%s[%s,%d features]'%(name,
                model_def.estimator.__class__.__name__, len(model_def.features))

    def _train(self, train_data):
        x, y, fitted_model = fit_model(self.model_def, train_data)
        return fitted_model

    def _apply(self, data, fitted_feature):
        fitted_model = fitted_feature.trained_data
        # if self.cv_folds:
        #     if isinstance(self.cv_folds, int):
        #         folds = make_folds(context.train_index, self.cv_folds)
        #     else:
        #         folds = self.cv_folds
        #     preds = []
        #     for train, test in folds:
        #         ctx = context.copy()
        #         ctx.train_index = train
        #         preds.append(self._predict(ctx, test, fit_model=True))
        #     # if there is held-out data, use all of train to predict
        #     # (these predictions use more data, so will be "better",
        #     # not sure if that is problematic...)
        #     remaining = context.data.index - context.train_index
        #     if len(remaining):
        #         preds.append(self._predict(context, remaining))
        #     preds = concat(preds, axis=0)
        # else:
        preds = self._predict(fitted_model, data)
        preds = DataFrame(preds, index=data.index)
        return preds

    def _predict(self, fitted_model, predict_data):
        x_test, y_true, y_preds = predict_model(self.model_def, predict_data, fitted_model)
        return y_preds

    def make_cross_validated_models(self, data, fitted_feature):
        pass


class Residuals(Predictions):

    def _predict(self, fitted_model, predict_data):
        x_test, y_true, y_preds = predict_model(self.model_def, predict_data, fitted_model)
        return y_preds - y_true


class FeatureSelector(ComboFeature):

    def __init__(self, features, selector, target, n_keep=50, train_only=True,
            cache=False):
        """ train_only: if true, features are selected only using training index data (recommended)"""
        super(FeatureSelector, self).__init__(features)
        self.selector = selector
        self.n_keep = n_keep
        self.target = target
        self.train_only = train_only
        self._cacheable = cache
        self._name = self._name + '_%d_%s'%(n_keep, selector.__class__.__name__)

    def depends_on_y(self):
        return self.train_only or super(FeatureSelector, self).depends_on_y()

    def _train(self, data):
        if self.train_only:
            y = get_single_column(self.target.create(self.context)).reindex(self.context.train_index)
            x = data.reindex(self.context.train_index)
        else:
            y = get_single_column(self.target.create(self.context))
            x = data
        cols = self.select(x, y)
        return cols

    def select(self, x, y):
        return self.selector.sets(x, y, self.n_keep)

    def _combine_apply(self, datas, fitted_feature):
        data = concat(datas, axis=1)
        cols = self.get_prep_data(data)
        return data[cols]


class TargetAggregationByFactor(Feature):
    """
    #TODO
    """
    def __init__(self, group_by, func=None, target=None, min_sample=10,
            verbose=False):
        # How terrible of a hack is this?
        feature = DummyFeature()
        super(TargetAggregationByFactor, self).__init__(feature)
        self.group_by = group_by
        self.func = func
        self.target = to_feature(target)
        self.min_sample = min_sample
        self.verbose = verbose

    def _train(self, train_data):
        y, ff = build_target_safe(self.target, train_data)
        vc = train_data[self.group_by].value_counts()
        keys = [k for k, v in vc.iterkv() if v >= self.min_sample]
        train_data['__grouping'] = train_data[self.group_by].map(lambda x: x if x in keys else '__other')
        train_data['__target'] = y
        vals = train_data.groupby('__grouping').agg({'__target': self.func})['__target'].to_dict()
        if self.verbose:
            print "\nPreparing Target Aggregations:"
            print vals.items()[:10]
        del train_data['__target']
        del train_data['__grouping']
        return (keys, vals)

    def _apply(self, data, fitted_feature):
        keys, vals = fitted_feature.trained_data
        if self.verbose:
            print "\nLoading Target aggs"
            print vals.items()[:10]
            print keys[:10]
            print data.columns
        data = data.applymap(lambda x: vals.get(x if x in keys else '__other'))
        return data
