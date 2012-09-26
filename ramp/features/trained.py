from base import ComboFeature, Feature, DummyFeature
from .. import models
# from ..core import Storable, store, get_key, get_dataset
from ..utils import make_folds
from pandas import Series, DataFrame, concat

# class ModelPredictions(Feature):
#     def __init__(self, config, name=None, cv_folds=5):
#         # TODO: Hackkkkk
#         super(ModelPredictions, self).__init__(config.features[0])
#         self.cv_folds = cv_folds
#         self.config = config
#         if name:
#             self._name = name
#         else:
#             self._name = '%s-%s' %(str(config.model)[:30], repr(self)[:8])

#     def _create(self, data):
#         return models.predict(self.dataset, self.config,
#                 self.dataset.data.index, self.train_index)

# class ModelResiduals(ModelPredictions):
#     def __init__(self, *args, **kwargs):
#         super(ModelResiduals, self).__init__(*args, **kwargs)

#     def _create(self, data):
#         preds = self.model.predict(self.dataset, self.features,
#             self.dataset.data.index,
#             self.dataset.train_index,
#             column_subset=self.column_subset,
#             target=self.target)
#         return self.dataset.get_train_y(target=self.target) - preds

class TrainedFeature(Feature):
    pass
    # def _create(self, data):
    #     if self.train_index is None:
    #         raise ValueError("A training index must be specified to create a "
    #         "TrainedFeature")
    #     return self.train(data)

    # def train(self, data):
    #     trainx = data.reindex(self.train_index)
    #     #testx = data.drop(self.train_index)
    #     y = self.dataset.get_train_y()
    #     trainy = y.reindex(self.train_index)
    #     self.fit(trainx, trainy)
    #     return self.predict(data)

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

    def __init__(self, features, selector, target, n_keep=50, trained=True,
            cache=False):
        super(FeatureSelector, self).__init__(features)
        self.selector = selector
        self.n_keep = n_keep
        self.target = target
        self.trained = trained
        self._cacheable = cache
        self._name = self._name + '_%d_%s'%(n_keep, selector.__class__.__name__)

    def is_trained(self):
        return self.trained

    def select(self, x, y):
        return self.selector.sets(x, y, self.n_keep)

    def combine(self, datas):
        data = concat(datas, axis=1)
        y = self.dataset.get_train_y(self.target, self.train_index)
        x = data.reindex(self.train_index)
        cols = self.select(x, y)
        return data[cols]


