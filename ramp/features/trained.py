from base import ComboFeature, Feature, DummyFeature, to_feature
from .. import models
from ..utils import make_folds, get_single_column
from pandas import Series, DataFrame, concat


class Predictions(Feature):
    # TODO: update for new context

    def __init__(self, config, name=None, external_context=None,
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
        self.config = config
        self.external_context = external_context
        self.feature = DummyFeature()
        self._cacheable = cache
        self.trained = True
        super(Predictions, self).__init__(self.feature)
        if self.external_context is not None:
            # Dont need to retrain if using external dataset to train
            self.trained = False
        if not name:
            name = 'Predictions'
        self._name = '%s[%s,%d features]'%(name,
                config.model.__class__.__name__, len(config.features))

    def depends_on_y(self):
        return self.trained

    def get_context(self):
        return self.external_context or self.context

    def _prepare(self, data):
        context = self.get_context()
        pre_data = context.data
        # only use training instances
        context.data = data.reindex(context.train_index)
        models.fit(self.config, context)
        context.data = pre_data
        return self.config.model

    def _create(self, data):
        context = self.get_context()
        if self.cv_folds:
            if isinstance(self.cv_folds, int):
                folds = make_folds(context.train_index, self.cv_folds)
            else:
                folds = self.cv_folds
            preds = []
            old_train_index = self.context.train_index
            old_prep_index = self.context.prep_index
            for train, test in folds:
                ctx = context.copy()
                self.context.train_index = train
                self.context.prep_index = train
                preds.append(self._predict(ctx, test))
            # if there is held-out data, use all of train to predict
            # (these predictions use more data, so will be "better",
            # not sure if that is problematic...)
            self.context.train_index = old_train_index
            self.context.prep_index = old_prep_index
            remaining = context.data.index - context.train_index
            if len(remaining):
                preds.append(self._predict(context, remaining))
            preds = concat(preds, axis=0)
        else:
            preds = self._predict(context)
        preds = DataFrame(preds)
        return preds

    def _predict(self, context, pred_index=None, fit_model=False):
        if pred_index is None:
            pred_index = context.data.index
        if not fit_model:
            model = self.get_prep_data(context.data)
            self.config.model = model
        return models.predict(self.config, context, pred_index, fit_model=fit_model)['predictions']


class Residuals(Predictions):

    def _predict(self, context, pred_index=None):
        if pred_index is None:
            pred_index = context.data.index
        preds = models.predict(self.config, context, pred_index)['predictions']
        return get_single_column(self.config.target.create(context)) - preds


class FeatureSelector(ComboFeature):

    def __init__(self, features, selector, target, n_keep=50, train_only=True,
            cache=False):
        """ train_only: if true, features are selected only using training index data (recommended)"""
        super(FeatureSelector, self).__init__(features)
        self.selector = selector
        self.n_keep = n_keep
        self.target = to_feature(target)
        self.train_only = train_only
        self._cacheable = cache
        self._name = self._name + '_%d_%s'%(n_keep, selector.__class__.__name__)

    def depends_on_y(self):
        return self.train_only or super(FeatureSelector, self).depends_on_y()

    def _prepare(self, data):
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

    def combine(self, datas):
        data = concat(datas, axis=1)
        cols = self.get_prep_data(data)
        return data[cols]


class TargetAggregationByFactor(Feature):
    """
    """
    def __init__(self, feature, func=None, target=None, min_sample=10,
            verbose=False):
        super(TargetAggregationByFactor, self).__init__(feature)
        self.func = func
        self.target = to_feature(target)
        self.min_sample = min_sample
        self.verbose = verbose

    def depends_on_y(self):
        return True

    def _prepare(self, data):
        y = get_single_column(self.target.create(self.context)).reindex(self.context.train_index)
        x = data.reindex(self.context.train_index)
        c = x.columns[0]
        vc = x[c].value_counts()
        keys = [k for k, v in vc.iterkv() if v > self.min_sample]
        x['__grouping'] = x[c].map(lambda x: x if x in keys else '__other')
        x['__target'] = y
        vals = x.groupby('__grouping').agg({'__target': self.func})['__target'].to_dict()
        if self.verbose:
            print "\nPreparing Target aggs"
            print vals.items()[:10]
        del x['__target']
        del x['__grouping']
        return (keys, vals)

    def _create(self, data):
        keys, vals = self.get_prep_data(data)
        if self.verbose:
            print "\nLoading Target aggs"
            print vals.items()[:10]
            print keys[:10]
            print data.columns
        data = data.applymap(lambda x: vals.get(x if x in keys else '__other', 0))
        return data
