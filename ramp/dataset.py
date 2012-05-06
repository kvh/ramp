from core import get_dataset, _register_dataset
from store import ShelfStore, DummyStore
from configuration import *
import core
from features import BaseFeature, Feature, ConstantFeature
from pandas import concat, DataFrame, Series, Index
import hashlib
from sklearn import cross_validation
from sklearn import feature_selection
import scipy
import numpy as np
import re
import random
import models
import copy
from utils import _pprint

__all__ = ['DataSet']


class DataSet(object):
    def __init__(self, data, name, validation_index=None, default_config=None,
            store=None):
        self.set_data(data)
        self.name = name
        self.default_config = default_config
        if validation_index is None:
            validation_index = Index([])
        self.validation_index = validation_index
        self.train_index = data.index - validation_index
        self._cache = {}
        # TODO: add dummy store and custom store
        if isinstance(store, basestring):
            self.store = ShelfStore(store)
        elif isinstance(store, ShelfStore):
            self.store = store
        else:
            self.store = DummyStore()
        # self.storable_hash = self._make_hash(
        #         self.name)
        _register_dataset(self)

    def set_data(self, data):
        self._data = data

    def __str__(self):
        return 'DataSet %s: %d x %d'%(self.name, len(self._data),
                len(self._data.columns))

    def load(self, key):
        return self._cache[key]
        # except KeyError:
        #     return self.store.load(key)

    def save(self, key, value):
        self._cache[key] = value
        # self.store.save(key, value)

    def make_feature(self, feature, train_index=None, force=False):
        if train_index is None:
            train_index = self.train_index
        # try:
        #     if force: raise KeyError
        #     d = self.load(feature.unique_name)
        # except KeyError:
        d = feature.create(self, train_index, force)
        # sanity check index is valid
        assert(not d.index - self._data.index)
        if not isinstance(feature, ConstantFeature):
            if any(d.std() < 1e-9):
                print "\n\nWARNING: Feature '%s' has constant column. \n\n" % feature.unique_name
        if np.isnan(d.values).any():
            # TODO HACK: this is not right
            if not feature.unique_name.startswith(
                    Configuration.DEFAULT_PREDICTIONS_NAME):
                print "\n\n***** WARNING: NAN in feature '%s' *****\n\n"%feature.unique_name
        #self.save(feature.unique_name, d)
        return d

    def get_x(self, features, train_index):
        # check for dupes
        colnames = set([f.unique_name for f in features])
        assert(len(features) == len(colnames))
        if not features:
            return
        x = []
        for feature in features:
            x.append(self.make_feature(feature, train_index))
        for d in x[:1]:
            assert((d.index == x[0].index).all())
        return concat(x, axis=1)

    def get_train_x(self, features, train_index=None):
        if train_index is None:
            train_index = self.train_index
        x = self.get_x(features, train_index)
        # if remove_constants:
        #     for col in x.columns:
        #         sd = x[col].std()
        #         if np.isnan(sd) or sd < 1e-9:
        #             print "WARNING: constant column '%s'"%col
                    #del x[col]
        return x.reindex(train_index)

    def get_validation_x(self, features):
        x = self.get_x(features, None)
        return x.reindex(self.validation_index)

    def get_y(self, target_feature, train_index=None):
        y = target_feature.create(self)
        # only support single target for now
        if isinstance(y, DataFrame):
            if len(y.columns) > 1:
                raise NotImplementedError("Multidimensional target not "
                "supported yet")
            return y[y.columns[0]]
        return y

    def get_train_y(self, target=None, index=None):
        if target is None:
            target = self.default_config.target
        y = self.get_y(target)
        if index is None:
            index = self.train_index
        if index is None:
            return y
        return y.reindex(index)

    def get_validation_data(self):
        return self._data.reindex(self.validation_index)

    def get_saved_configurations(self, **kwargs):
        try:
            saved = self.load('saved_scores')
        except KeyError:
            saved = []
        if kwargs:
            return filter(lambda x: x[1].match(**kwargs), saved)
        return saved

    def save_configurations(self, configs):
        """configs: list of (cv_scores, config) """
        saved = self.get_saved_configurations()
        saved.extend(configs)
        self.dump('saved_scores', saved) #saved[:self.keep_nmodels])

    # def get_best_configuration(self, rank=0, weight_func=None, **kwargs):
    #     saved = self.get_saved_configurations()
    #     metric = saved[0][1].metric
    #     if weight_func is None:
    #         saved.sort(reverse=metric.reverse, key=lambda x:x[0].mean())
    #     else:
    #         saved.sort(reverse=metric.reverse, key=lambda x:weight_func(x[0]))
    #     return saved[rank]

    def print_models(self, n=10, q=None, show_cols=False, **kwargs):
        saved = self.get_saved_configurations(**kwargs)
        metrics = []
        seen = set()
        for s, conf in saved:
            if conf.metric.__class__ not in seen:
                metrics.append(conf.metric)
                seen.add(conf.metric.__class__)
        shown = []
        for metric in metrics:
            models = [(scores.mean(), scores.std(), len(config.column_subset), config.model,
                config.target, config.column_subset, config) for
                    scores, config in saved if config.metric.__class__ == metric.__class__]
            models.sort(reverse=metric.reverse)
            shown = [m[-1] for m in models]
            print metric.__class__.__name__
            print "\tmean\tstd\t# features\tmodel\ttarget"
            for i, m in enumerate(models[:n]):
                print str(i) + "\t%0.3f\t%0.3f\t%d\t%s\t%50r" % m[:-2]
                if show_cols:
                    print m[-2]
        return shown


    def pick_config(self, n=50):
        all_models = self.get_saved_models(**{})
        # metrics_ = list(set([c.metric.__class__ for s, c in all_models]))
        # targets = list(set([c.target.unique_name for s, c in all_models]))
        # models_ = list(set([c.model.__class__ for s, c in all_models]))
        # if not metrics_: return None
        # def enum_print(x):
        #     for i, m in enumerate(x):
        #         print i, m
        # if len(metrics_) > 1:
        #     enum_print(metrics_)
        #     metric = metrics_[int(raw_input('> '))]
        # else:
        #     metric = metrics_[0]
        # if len(targets) > 1:
        #     enum_print(targets)
        #     target = targets[int(raw_input('> '))]
        # else:
        #     target = targets[0]
        # if len(models_) > 1:
        #     enum_print(models_)
        #     model = models_[int(raw_input('> '))]
        # else:
        #     model = models_[0]
        models = self.print_models(n)#, metric=metric(1), target_name=target, model=model())
        print "Select a model"
        inp = raw_input('> ')
        if inp:
            return models[int(inp)]
        return None

    def explore(self, config, selector, start=1, end=None, stepsize=1, repeat=1, save_top_n=3):
        if config is None:
            config = self.default_config
        # x = self.get_train_x(config.features)
        # y = self.get_train_y(config.target)
        # xv = x.values
        # yv = y.values

        # print x
        # print y.describe()
        fitmodels = []
        sets = selector.sets_config(self, config)
        from_cols = config.column_subset
        if from_cols:
            best_cols=from_cols
            new_sets = [best_cols]
            cur = best_cols
            for s in sets:
                for i in set(s) - set(cur):
                    cur = cur + [i]
                    new_sets.append(cur)
            sets = new_sets
        if end is None:
            end = len(x.columns)
        # sets = [sets[10]] * 30
        stepstart = start % stepsize
        try:
            for i, col_names in enumerate(sets):
                if i < start: continue
                if i > end: break
                if i % stepsize != stepstart: continue
                #feature_set = [f for f in features if f.unique_name in col_names]
                if not col_names: continue
                nfeats = len(col_names)
                config.column_subset = col_names
                # print col_names
                print "\nAccuracy for '%d' features"%nfeats
                scores = models.cv(self, config,
                        folds=5, repeat=repeat)
                print "%s: %0.3f (+/- %0.3f) [%0.3f,%0.3f]" % (str(config.model)[:50],
                          scores.mean(), scores.std(), min(scores),
                          max(scores))
        #         skmodel = model.skmodel
                fitmodels.append((scores, copy.copy(config)))
        #         if hasattr(skmodel, 'coef_') and len(x.columns) < 30:
        #             for f, cf in zip(skmodel.coef_, x.columns):
        #                 print '%s\t%0.3f' % c[:100], cf
        finally:
            print "saving models"
            fitmodels.sort(reverse=config.metric.reverse, key=lambda x: x[0].mean())
            self.save_models(fitmodels[:save_top_n])


