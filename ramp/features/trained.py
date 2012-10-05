from base import ComboFeature, Feature, DummyFeature
from .. import models
# from ..core import Storable, store, get_key, get_dataset
from ..utils import make_folds
from pandas import Series, DataFrame, concat
from ..dataset import get_single_column


class Predictions(Feature):

    def __init__(self, config, name=None, cv_folds=5,
            external_dataset_name=None, cache=False):
        self.cv_folds = cv_folds
        self.config = config
        self.external_dataset_name=external_dataset_name
        self.feature = DummyFeature()
        self._cacheable = cache
        self.trained = True
        super(Predictions, self).__init__(self.feature)
        if self.external_dataset_name is not None:
            # Dont need to retrain if using external dataset to train
            self.trained = False
        if not name:
            name = 'Predictions'
        self._name = '%s[%s,%d features]'%(name,
                config.model.__class__.__name__, len(config.features))

    def is_trained(self):
        return self.trained

    def set_train_index(self, index):
        self.train_index = index

    def _create(self, data):
        self.train_dataset = self.dataset
        if not hasattr(self, 'train_index') or self.train_index is None:
            print "No training index provided, using dataset default."
            self.train_index = self.dataset.train_index
        if self.external_dataset_name:
            self.train_dataset = get_dataset(self.external_dataset_name)
            self.train_index = self.train_dataset.train_index
        # print "building predictions on %d training samples (%d total) from dataset '%s'"%(
        #         len(self.train_index), len(self.dataset.data), self.train_dataset.name)
        # print "train index:", self.train_index[:20]
        preds = self._predict()
        preds = DataFrame(preds)
        return preds

    def _predict(self):
        return models.predict(self.dataset, self.config,
                self.dataset._data.index, self.train_index, self.train_dataset)


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

