  # -*- coding: utf-8 -*-
'''
Features: Base
-------

The Base Features module provides a set of abstract base classes and simple
feature definitions (such as Length, Log, Power, etc). The ABC's can be built
upon, such as in the text or combo modules. 

These Feature classes allow for chaining and combining feature sets to be 
iterated upon in the Configurator Factory. 

'''
from hashlib import md5
import math
import random
import re

import numpy as np
from pandas import Series, DataFrame, concat

from ramp.store import Storable
from ramp.utils import (_pprint, get_np_hashable, key_from_index,
                        get_single_column, stable_repr, reindex_safe)


available_features = []


class FittedFeature(Storable):

    def __init__(self, feature, train_index, prep_index, prepped_data=None,
                 trained_data=None, inner_fitted_features=None):
        self.feature = feature

        # compute metadata
        self.train_n = len(train_index)
        self.prep_n = len(prep_index)
        self.train_data_key = key_from_index(train_index)
        self.prep_data_key = key_from_index(prep_index)

        self.inner_fitted_features = inner_fitted_features

        self.prepped_data = prepped_data
        self.trained_data = trained_data


class BaseFeature(object):
    """
    BaseFeature wraps a string corresponding to a
    DataFrame column.
    """

    _cacheable = True

    def __init__(self, feature):
        if not isinstance(feature, basestring):
            raise ValueError('Base feature must be a string')
        self.feature = feature

    def __repr__(self):
        return repr(self.feature)

    def __str__(self):
        return str(self.feature)

    @property
    def unique_name(self):
        return str(self)

    def depends_on_y(self):
        return False

    def depends_on_other_x(self):
        return False

    @property
    def is_trained(self):
        return self.depends_on_y()

    @property
    def is_prepped(self):
        return self.depends_on_other_x()

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def build(self, data, prep_index=None, train_index=None):
        feature_data = self.apply(data)
        return feature_data, None

    def apply(self, data, fitted_feature=None):
        return DataFrame(data[self.feature],
                         columns=[self.feature])

    def prepare(self, prep_data):
        raise NotImplementedError

    def train(self, train_data):
        raise NotImplementedError

    def __add__(self, other):
        return combo.Add([self, other])

    def __sub__(self, other):
        return combo.Sub([self, other])

    def __div__(self, other):
        return combo.Divide([self, other])

    def __mul__(self, other):
        return combo.Multiply([self, other])

    def __pow__(self, power):
        return Power(self, power)


class ConstantFeature(BaseFeature):

    def __init__(self, feature):
        if not isinstance(feature, int) and not isinstance(feature, float):
            raise ValueError('Constant feature must be a number')
        self.feature = feature

    def apply(self, data, fitted_feature=None):
        return DataFrame(np.repeat(self.feature, len(data)),
                         index=data.index,
                         columns=['%s' % self.feature])


class DummyFeature(BaseFeature):
    """ For testing """

    def __init__(self):
        self.feature = ''

    def apply(self, data, fitted_feature=None):
        return data
AllDataFeature = DummyFeature


class FeatureMetaClass(type):
    def __new__(meta, name, bases, dct):
        available_features.append(name)
        return super(FeatureMetaClass, meta).__new__(meta, name, bases, dct)

    def __call__(cls, *args, **kwargs):
        for arg in list(args) + kwargs.values():
            if hasattr(arg, '__call__') and getattr(arg, '__name__', 0) == (lambda: None).__name__:
                Warning("Feature's should not be passed anonymous (lambda) functions"
                        " as these cannot be serialized reliably")
        return type.__call__(cls, *args, **kwargs)


class ComboFeature(BaseFeature):
    """
    Abstract base for more complex features
    """

    __metaclass__ = FeatureMetaClass

    hash_length = 8
    _cacheable = True
    re_hsh = re.compile(r' \[\w{%d}\]' % hash_length)

    def __init__(self, features):
        """
        Inheriting classes responsible for setting human-readable description of
        feature and parameters on _name attribute.
        """
        self.features = []
        # handle single feature as well
        if not isinstance(features, list) and not isinstance(features, tuple):
            features = [features]
        for feature in features:
            if isinstance(feature, basestring):
                feature = BaseFeature(feature)
            if isinstance(feature, int) or isinstance(feature, float):
                feature = ConstantFeature(feature)
            self.features.append(feature)
        self.set_name()

    def set_name(self):
        cname = self.__class__.__name__
        if cname.endswith('Feature'):
            cname = cname[:-7]
        # _name attribute is for human-readable strings
        self._name = cname

    def __getstate__(self):
        # shallow copy dict and keep references
        dct = self.__dict__.copy()
        return dct

    def __repr__(self):
        return stable_repr(self)

    def _hash(self):
        s = repr(self)
        return md5(s).hexdigest()[:self.hash_length]

    @property
    def unique_name(self):
        """
        Must provide a unique string as a funtion of this feature, its
        parameter settings, and all it's contained features. It should also be
        readable and maintain a reasonable length (by hashing, for instance).
        """
        h = self._hash()
        return '%s [%s]' %(self, h)

    def __str__(self):
        """
        A readable version of this feature (and its contained features).
        Should be as short as possible.
        """
        f = ', '.join([str(f) for f in self.features])
        if self._name:
            return '%s(%s)' %(self._name, f)
        # else this is just a Feature wrapper, no need to add anything
        return f

    def _remove_hashes(self, s):
        return self.re_hsh.sub('', s)

    def column_rename(self, existing_name, hsh=None):
        """
        Like unique_name, but in addition must be unique to each column of this
        feature. accomplishes this by prepending readable string to existing
        column name and replacing unique hash at end of column name.
        """
        try:
            existing_name = str(existing_name)
        except UnicodeEncodeError:
            pass
        if hsh is None:
            hsh = self._hash()
        if self._name:
            return '%s(%s) [%s]' %(self._name, self._remove_hashes(existing_name),
                    hsh)
        return '%s [%s]'%(self._remove_hashes(existing_name),
                    hsh)

    def depends_on_y(self):
        if hasattr(self, '_train'):
            return True
        return any([f.depends_on_y() for f in self.features])

    def depends_on_other_x(self):
        if hasattr(self, '_prepare'):
            return True
        return any([f.depends_on_other_x() for f in self.features])

    def build(self, data, prep_index=None, train_index=None):
        if prep_index is None:
            prep_index = data.index
        if train_index is None:
            train_index = data.index
        datas = []
        fitted_features = []
        for feature in self.features:
            feature_data, ff = feature.build(data, prep_index, train_index)
            datas.append(feature_data)
            fitted_features.append(ff)
        ff = FittedFeature(self,
                           prep_index=prep_index,
                           train_index=train_index,
                           inner_fitted_features=fitted_features)
        ff.prepped_data = self.prepare([reindex_safe(d, prep_index) for d in datas])
        ff.trained_data = self.train([reindex_safe(d, train_index) for d in datas])
        feature_data = self._combine_apply(datas, ff)
        feature_data = self._prepend_feature_name_to_all_columns(feature_data)
        return feature_data, ff

    def _prepend_feature_name_to_all_columns(self, data):
        hsh = self._hash() # cache this so we dont recompute for every column
        data.columns = [self.column_rename(c, hsh) for c in data.columns]
        return data

    def apply(self, data, fitted_feature):
        datas = []
        for feature, inner_fitted_feature in zip(self.features, fitted_feature.inner_fitted_features):
            datas.append(feature.apply(data, inner_fitted_feature))
        feature_data = self._combine_apply(datas, fitted_feature)
        if not isinstance(feature_data, DataFrame):
            raise TypeError("_combine_apply() method must return a DataFrame")
        return self._prepend_feature_name_to_all_columns(feature_data)

    def _apply(self, data, fitted_feature):
        raise NotImplementedError

    def _combine_apply(self, datas, fitted_feature):
        raise NotImplementedError

    def prepare(self, prep_datas):
        if hasattr(self, '_prepare'):
            prepped_data = self._prepare(prep_datas)
            return prepped_data
        else:
            return None

    def train(self, train_datas):
        if hasattr(self, '_train'):
            trained_data = self._train(train_datas)
            return trained_data
        else:
            return None


class Feature(ComboFeature):
    """
    Base class for features operating on a single feature.
    """
    def __init__(self, feature):
        super(Feature, self).__init__([feature])
        self.feature = self.features[0]

    # def apply(self, data, fitted_feature):
    #     # recurse:
    #     data = self.feature.apply(data, fitted_feature.inner_fitted_feature)
    #     # apply this feature's transformation:
    #     feature_data = self._apply(data, fitted_feature)
    #     if not isinstance(feature_data, DataFrame):
    #         raise TypeError("_apply() method must return a DataFrame")
    #     return self._prepend_feature_name_to_all_columns(feature_data)

    def _apply(self, data, fitted_feature):
        return data

    def _combine_apply(self, datas, fitted_feature):
        return self._apply(datas[0], fitted_feature)

    def prepare(self, prep_datas):
        if hasattr(self, '_prepare'):
            prepped_data = self._prepare(prep_datas[0])
            return prepped_data
        else:
            return None

    def train(self, train_datas):
        if hasattr(self, '_train'):
            trained_data = self._train(train_datas[0])
            return trained_data
        else:
            return None
# shortcut
F = Feature


class MissingIndicator(Feature):
    """
    Adds a missing indicator column for this feature.
    Indicator will be 1 if given feature `isnan` (numpy definition), 0 otherwise.
    """
    def _apply(self, data, fitted_feature):
        for col in data.columns:
            missing = data[col].map(lambda x: int(x.isnan()))
            missing.name = 'missing_%s'%col
            data.append(missing)
        return data


class FillMissing(Feature):
    """
    Fills `na` values (pandas definition) with `fill_value`.
    """
    def __init__(self, feature, fill_value):
        self.fill_value = fill_value
        super(FillMissing, self).__init__(feature)

    def _apply(self, data, fitted_feature):
        return data.fillna(self.fill_value)


class MissingIndicatorAndFill(Feature):
    """
    Adds a missing indicator column for this feature.
    Indicator will be 1 if given feature `isnan` (numpy definition), 0 otherwise,
    and then fill NaNs with `fill_value`.
    """
    def __init__(self, feature, fill_value):
        self.fill_value = fill_value
        super(MissingIndicatorAndFill, self).__init__(feature)

    def _apply(self, data, fitted_feature):
        cols = []
        names = []
        for col in data.columns:
            missing = data[col].map(lambda x: int(np.isnan(x)))
            names.append( 'missing_%s'%col)
            cols.append(missing)
        data = concat([data, concat(cols, keys=names, axis=1)], axis=1)
        return data.fillna(self.fill_value)


# class DropConstant(ComboFeature):
#     def _prepare(self, data):
#         dropped_cols = []
#         for col in data.columns:
#             if data[col].std() < 1e-9:
#                 dropped_cols.append(col)
#         print "Dropped %d columns", (len(dropped_cols), dropped_cols)
#         return {"dropped_columns": dropped_cols}

#     def combine(self, datas):
#         data = concat(datas, axis=1)        
#         cols = fitted_feature.prepped_data['dropped_columns']
#         return data.drop(cols, axis=1)


class Length(Feature):
    """
    Applies builtin `len` to feature.
    """
    def _apply(self, data, fitted_feature):
        return data.applymap(lambda x: len(x))


class Normalize(Feature):
    """
    Normalizes feature to mean zero, stdev one.
    """
    def _prepare(self, data):
        cols = {}
        for col in data.columns:
            d = data[col]
            m = d.mean()
            s = d.std()
            cols[col] = (m, s)
        return cols

    def _apply(self, data, fitted_feature):
        eps = 1.0e-10
        col_stats = fitted_feature.prepped_data
        d = DataFrame(index=data.index)
        for col in data.columns:
            m, s = col_stats.get(col, (0, 0))
            if s < eps:
                d[col] = data[col] - m
            else:
                d[col] = (data[col] - m) / s
        return d


class Discretize(Feature):
    """
    Bins values based on given cutoffs.
    """
    def __init__(self, feature, cutoffs, values=None):
        super(Discretize, self).__init__(feature)
        self.cutoffs = cutoffs
        if values is None:
            values = range(len(cutoffs) + 1)
        self.values = values

    def discretize(self, x):
        for i, c in enumerate(self.cutoffs):
            if x < c:
                return self.values[i]
        return self.values[-1]

    def _apply(self, data, fitted_feature):
        return data.applymap(self.discretize)


class Map(Feature):
    """
    Applies given function to feature. Feature *cannot*
    be anonymous (ie lambda). Must be defined in top level
    (and thus picklable).
    """
    def __init__(self, feature, function, name=None):
        super(Map, self).__init__(feature)
        self.function = function
        if name is None:
            name = function.__name__
        self.name = name
        self._name = name

    def _apply(self, data, fitted_feature):
        return data.applymap(self.function)


class AsFactor(Feature):
    """
    Maps nominal values to ints and stores
    mapping. Mapping may be provided at definition.
    """
    def __init__(self, feature, levels=None):
        """ levels is list of tuples """
        super(AsFactor, self).__init__(feature)
        self.levels = levels

    def _prepare(self, data):
        levels = self.levels
        if not levels:
            levels = set(get_single_column(data))
            levels = zip(levels, range(len(levels)))
        return levels

    def _apply(self, data, fitted_feature):
        levels = fitted_feature.prepped_data
        mapping = dict(levels)
        return data.applymap(mapping.get)

    def get_name(self, factor):
        factors = self.get_prep_data()
        inverse = dict([(v,k) for k,v in factors])
        return inverse.get(factor)


class AsFactorIndicators(Feature):
    """
    Maps nominal values to indicator columns. So
    a column with values ['good', 'fair', 'poor'],
    would be mapped to three indicator columns
    if include_all is True otherwise two columns (the
    third implied by zeros on the other two columns)
    """
    def __init__(self, feature, levels=None, include_all=True):
        super(AsFactorIndicators, self).__init__(feature)
        self.levels = levels
        self.include_all = include_all

    def _prepare(self, data):
        levels = self.levels
        if not levels:
            levels = sorted(set(get_single_column(data)))
        return levels

    def _apply(self, data, fitted_feature):
        factors = fitted_feature.prepped_data
        data = get_single_column(data)
        d = DataFrame(index=data.index)
        if self.include_all:
            facts = list(factors)
        else:
            facts = list(factors)[:-1]
        for f in facts:
            d['%s-%s'%(f, data.name)] = data.map(lambda x: int(x == f))
        return d


class IndicatorEquals(Feature):
    """
    Maps feature to one if equals given value, zero otherwise.
    """
    def __init__(self, feature, value):
        super(IndicatorEquals, self).__init__(feature)
        self.value = value
        self._name = self._name + '_%s'%value

    def _apply(self, data, fitted_feature):
        return data.applymap(lambda x: int(x==self.value))


class Log(Map):
    """
    Takes log of given feature. User is responsible for
    ensuring values are in domain.
    """
    def __init__(self, feature):
        super(Log, self).__init__(feature, math.log)


class Power(Feature):
    """
    Takes feature to given power. Equivalent to operator: F('a') ** power.
    """
    def __init__(self, feature, power=2):
        self.power = power
        super(Power, self).__init__(feature)

    def _apply(self, data, fitted_feature):
        return data.applymap(lambda x: x ** self.power)


class GroupMap(Feature):
    """
    Applies a function over specific sub-groups of the data
    Typically this will be with a MultiIndex (hierarchical index).
    If group is encountered that has not been seen, defaults to
    global map.
    TODO: prep this feature
    """

    def __init__(self, feature, function, name=None, **groupargs):
        super(GroupMap, self).__init__(feature)
        self.function = function
        if name is None:
            name = function.__name__
        self.name = name
        self._name = name
        self.groupargs = groupargs

    def _apply(self, data, fitted_feature):
        d = data.groupby(**self.groupargs).applymap(self.function)
        return d


class GroupAggregate(ComboFeature):
    """
    Computes an aggregate value by group.

    Groups can be specified with kw args which will be
    passed to the pandas `groupby` method, or by
    specifying a `groupby_column` which will group by value
    of that column.
    """
    def __init__(self, features, function, name=None, data_column=None,
            trained=False, groupby_column=None, **groupargs):
        super(GroupAggregate, self).__init__(features)
        self.function = function
        self.data_column = data_column
        self.trained = trained
        self.groupby_column = groupby_column
        self._name = name or function.__name__
        self.groupargs = groupargs

    def depends_on_y(self):
        return self.trained or super(GroupAggregate, self).depends_on_y()

    def _prepare(self, data):
        prep = {}
        # global value
        prep['global'] = self.function(data[self.data_column])
        # group specific
        if self.groupby_column:
            g = data.groupby(by=self.groupby_column, **self.groupargs)
        else:
            g = data.groupby(**self.groupargs)
        d = g[self.data_column].apply(self.function)
        prep['groups'] = d
        return prep

    def _create(self, datas):
        data = concat(datas, axis=1)
        prep = fitted_feature.prepped_data
        globl = prep['global']
        groups = prep['groups']
        if self.groupby_column:
            data = DataFrame(data[self.groupby_column].apply(lambda x: groups.get(x, globl)))
        else:
            data = DataFrame([groups.get(x, globl) for x in data.index],
                    columns=[self.data_column],
                    index=data.index)
        return data


def contain(x, mn, mx):
    if mx is not None and x > mx: return mx
    if mn is not None and x < mn: return mn
    return x

class Contain(Feature):
    """
    Trims values to inside min and max.
    """
    def __init__(self, feature, min=None, max=None):
        self.min = min
        self.max = max
        super(Contain, self).__init__(feature)
        self._name = self._name + '(%s,%s)' %(min, max)

    def _apply(self, data, fitted_feature):
        return data.applymap(lambda x: contain(x, self.min, self.max))


class ReplaceOutliers(Feature):
    # TODO: add prep
    def __init__(self, feature, stdevs=7, replace='mean'):
        super(ReplaceOutliers, self).__init__(feature)
        self.stdevs = stdevs
        self.replace = replace
        self._name = self._name + '_%d'%stdevs

    def is_outlier(self, x, mean, std):
        return abs((x-mean)/std) > self.stdevs

    def _apply(self, data, fitted_feature):
        replace = self.replace
        cols = []
        for col in data.columns:
            d = data[col]
            m = d.mean()
            s = d.std()
            if self.replace == 'mean':
                replace = m
            cols.append(d.map(lambda x: x if not self.is_outlier(x, m, s) else
                    replace)
                    )
        return concat(cols, keys=data.columns, axis=1)


class ColumnSubset(Feature):
    def __init__(self, feature, subset, match_substr=False):
        super(ColumnSubset, self).__init__(feature)
        self.subset = subset
        self.match_substr = match_substr

    def _apply(self, data, fitted_feature):
        if self.match_substr:
            cols = [c for c in data.columns if any([s in c for s in self.subset])]
        else:
            cols = self.subset
        return data[cols]


class Lag(Feature):
    """
    Lags given feature by n along index.
    """
    def __init__(self, feature, lag=1, fill=0):
        self.lag = lag
        self.fill = fill
        super(Lag, self).__init__(feature)
        self._name = self._name + '_%d'%self.lag

    def _apply(self, data, fitted_feature):
        cols = []
        for col in data.columns:
            cols.append(Series([self.fill] * self.lag + list(data[col][:-self.lag]),
                            index=data.index))
        return concat(cols, keys=data.columns, axis=1)


def to_feature(feature_like):
    return feature_like if isinstance(feature_like, BaseFeature) else Feature(feature_like)


import combo

