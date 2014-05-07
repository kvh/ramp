import logging
from pandas import Series, DataFrame, concat

from ramp.builders import build_target_safe
from ramp.features.base import to_feature, ComboFeature, Feature, AllDataFeature
from ramp.modeling import fit_model, generate_test
from ramp.utils import make_folds, get_single_column, reindex_safe


class TrainedFeature(Feature):

    def __init__(self):
        # For trained features, we will need access to all the data
        self.feature = AllDataFeature()
        super(TrainedFeature, self).__init__(self.feature)


class Predictions(TrainedFeature):
    # TODO: update for new context

    def __init__(self, model_def, name=None, external_data=None,
            cv_folds=None):
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
        self.external_data = external_data
        super(Predictions, self).__init__()
        #TODO
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

        #TODO
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
        x_test, y_true = generate_test(self.model_def, predict_data, fitted_model)
        y_preds = self.model_def.estimator.predict(x_test)
        return y_preds

    def make_cross_validated_models(self, data, fitted_feature):
        pass


class Residuals(Predictions):

    def _predict(self, fitted_model, predict_data):
        x_test, y_true = generate_test(self.model_def, predict_data, fitted_model)
        y_preds = self.model_def.estimator.predict(x_test)
        return y_preds - y_true


class FeatureSelector(ComboFeature):

    def __init__(self, features, selector, target, data, n_keep=50,
                 threshold_arg=None):
        """
        """
        super(FeatureSelector, self).__init__(features)
        self.selector = selector
        self.n_keep = n_keep
        self.threshold_arg = threshold_arg
        self.target = target
        self.data = data
        self._name = self._name + '_%d_%s'%(threshold_arg or n_keep, selector.__class__.__name__)

    def _train(self, train_datas):
        train_data = concat(train_datas, axis=1)
        y, ff = build_target_safe(self.target, self.data)
        y = reindex_safe(y, train_data.index)
        arg = self.threshold_arg
        if arg is None:
            arg = self.n_keep
        cols = self.selector.select(train_data, y, arg)
        return cols

    def _combine_apply(self, datas, fitted_feature):
        data = concat(datas, axis=1)
        selected_columns = fitted_feature.trained_data
        return data[selected_columns]


class TargetAggregationByFactor(TrainedFeature):
    """
    """
    def __init__(self, group_by, func=None, target=None, min_sample=10):
        # How terrible of a hack is this?
        super(TargetAggregationByFactor, self).__init__()
        self.group_by = group_by
        self.func = func
        self.target = to_feature(target)
        self.min_sample = min_sample

    def _train(self, train_data):
        y, ff = build_target_safe(self.target, train_data)
        vc = train_data[self.group_by].value_counts()
        keys = [k for k, v in vc.iterkv() if v >= self.min_sample]
        train_data['__grouping'] = train_data[self.group_by].map(lambda x: x if x in keys else '__other')
        train_data['__target'] = y
        vals = train_data.groupby('__grouping').agg({'__target': self.func})['__target'].to_dict()
        logging.debug("Preparing Target Aggregations:")
        logging.debug(str(vals.items()[:10]))
        del train_data['__target']
        del train_data['__grouping']
        return (keys, vals)

    def _apply(self, data, fitted_feature):
        keys, vals = fitted_feature.trained_data
        logging.debug("Loading Target aggs")
        logging.debug(str(vals.items()[:10]))
        logging.debug(str(keys[:10]))
        logging.debug(str(data.columns))
        data = data.applymap(lambda x: vals.get(x if x in keys else '__other'))
        return data
