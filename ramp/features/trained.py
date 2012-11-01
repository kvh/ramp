from base import ComboFeature, Feature, DummyFeature
from .. import models
from ..utils import make_folds, get_single_column
from pandas import Series, DataFrame, concat


class Predictions(Feature):
    # TODO: update for new context

    def __init__(self, config, name=None, external_context=None,
            cv_folds=5, cache=False):
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

    def depends_on_other_x(self):


    def _create(self, data):
        context = self.context
        if self.external_context:
            context = self.external_context
        preds = self._predict(context)
        preds = DataFrame(preds)
        return preds

    def _predict(self, context):
        return models.predict(self.config, context)


class Residuals(Predictions):

    def _predict(self):
        preds = models.predict(self.dataset, self.config,
                self.dataset.train_index, self.train_index, self.train_dataset)
        return self.dataset.get_train_y(target=self.config.actual) - preds


class CVPredictions(Predictions):

    def _predict(self):
        if self.train_index is None:
            raise ValueError("A training index must be specified to create a "
            "TrainedFeature")
        preds = []
        # the actual predictions made are from k-fold cross val
        # otherwise the model will fit better in training than in
        # validation/test (you may end up with nested cross-val here...)
        for train, test in make_folds(self.train_index, self.cv_folds):
            # print train, test
            preds.append(models.predict(self.dataset, self.config, test, train,
                self.train_dataset))
        if self.train_dataset == self.dataset:
            valid_index = self.dataset._data.index - self.train_index
        else:
            valid_index = self.dataset._data.index
        if len(valid_index):
            preds.append(models.predict(self.dataset, self.config,
                valid_index,
                self.train_index,
                self.train_dataset
                ))
        data = concat(preds, axis=0)
        return data


class FeatureSelector(ComboFeature):

    def __init__(self, features, selector, target, n_keep=50, train_only=True,
            cache=False):
        super(FeatureSelector, self).__init__(features)
        self.selector = selector
        self.n_keep = n_keep
        self.target = target
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

