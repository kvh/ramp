from pandas import Series, DataFrame, concat
import numpy as np
import random
import inspect
import math
import re
from hashlib import md5
re_object_repr = re.compile(r'<([.a-zA-Z0-9_ ]+?)\sat\s\w+>')
from ..utils import _pprint, get_hash


def get_single_column(df):
    assert(len(df.columns) == 1)
    return df[df.columns[0]]


class BaseFeature(object):

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

    def is_trained(self):
        return False

    def set_train_index(self, index):
        pass

    def create(self, dataset, *args, **kwargs):
        return DataFrame(dataset._data[self.feature].copy(),
                columns=[self.feature])

    def __add__(self, other):
        return combo.Add([self, other])

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

    def create(self, dataset, *args, **kwargs):
        return DataFrame(
                [self.feature] * len(dataset._data),
                index=dataset._data.index,
                columns=['%s'%self.feature])


class DummyFeature(BaseFeature):

    def __init__(self):
        self.feature = ''

    def create(self, dataset, *args, **kwargs):
        return dataset._data


class ComboFeature(BaseFeature):

    hash_length = 8
    _cacheable = True

    def __init__(self, features):
        self.features = []
        if not isinstance(features, list) and not isinstance(features, tuple):
            features = [features]
        for feature in features:
            if isinstance(feature, basestring):
                feature = BaseFeature(feature)
            if isinstance(feature, int) or isinstance(feature, float):
                feature = ConstantFeature(feature)
            self.features.append(feature)
        cname = self.__class__.__name__
        if cname.endswith('Feature'):
            cname = cname[:-7]
        self._name = cname

    def __getstate__(self):
        # shallow copy dict so we don't modify references
        dct = self.__dict__.copy()
        # HACK remove ephemeral items
        if 'dataset' in dct:
            del dct['dataset']
        if 'train_index' in dct:
            del dct['train_index']
        if 'train_dataset' in dct:
            del dct['train_dataset']
        return dct

    def __repr__(self):
        state = _pprint(self.__getstate__())
        # HACK: replace 'repr's that contain object id references
        state = re_object_repr.sub(r'<\1>', state)
        return '%s(%s)' % (
                self.__class__.__name__,
                state)

    def _hash(self):
        s = repr(self)
        if hasattr(self, 'train_index') and self.train_index is not None:
            s += '-' + get_hash(self.train_index)
        return md5(s).hexdigest()[:self.hash_length]

    @property
    def unique_name(self):
        """
        must provide a unique string as a funtion of this feature, its
        parameter settings, and all it's contained features. It should also be
        readable and maintain a reasonable length (by hashing, for instance).
        """
        h = self._hash()
        return '%s [%s]' %(self, h)

    def save(self, key, value):
        self.dataset.save('%s-%s'%(self.unique_name, key), value)
    def load(self, key):
        return self.dataset.load('%s-%s'%(self.unique_name, key))

    def __str__(self):
        """
        a readable version of this feature (and its contained features)
        """
        f = ', '.join([str(f) for f in self.features])
        if self._name:
            return '%s(%s)' %(self._name, f)
        # else this is just a Feature wrapper, no need to add anything
        return f

    def _remove_hashes(self, s):
        return re.sub(r' \[\w{%d}\]'%self.hash_length, '', s)

    def column_rename(self, existing_name):
        """
        like unique_name, but in addition must be unique to each column of this
        feature. accomplishes this by prepending readable string to existing
        column name and replacing unique hash at end of column name.
        """
        try:
            existing_name = str(existing_name)
        except UnicodeEncodeError:
            pass
        if self._name:
            return '%s(%s) [%s]' %(self._name, self._remove_hashes(existing_name),
                    self._hash())
        return '%s [%s]'%(self._remove_hashes(existing_name),
                    self._hash())
        # if self._name:
        #     return '%s(%s)' %(self._name, existing_name
        #             )
        # return '%s'%(existing_name)

    def is_trained(self):
        return any([f.is_trained() for f in self.features])

    # def set_train_index(self, index):
    #     self.train_index = index
    #     for feature in self.features:
    #         feature.set_train_index(index)

    def create(self, dataset, train_index=None, force=False):
        """ This is the prep for creating features. Has caching logic. """
        self.dataset = dataset
        if self.is_trained():
            self.train_index = train_index
        try:
            if force: raise KeyError
            d = self.dataset.store.load(self.unique_name)
            print "loading '%s' for dataset '%s'" % (self.unique_name,
                self.dataset.name)
            return d
        except KeyError:
            print "creating '%s' for dataset '%s'..." % (self.unique_name,
                self.dataset.name),
            pass
        datas = []

        # recurse
        for feature in self.features:
            data = feature.create(dataset, train_index, force)
            # copy the dataframe to isolate side effects
            # TODO: is this really necessary? Can we enforce immutability?
            data = DataFrame(data.copy())
            datas.append(data)

        # actually apply the feature
        data = self._create(datas)

        # cache it
        if self._cacheable:
            self.dataset.store.save(self.unique_name, data)

        # delete state attrs. features are stateless!
        if self.is_trained():
            del self.train_index
        if hasattr(self, 'train_dataset'):
            del self.train_dataset
        print "done"
        return data

    def _create(self, datas):
        data = self.combine(datas)
        data.columns = data.columns.map(self.column_rename)
        return data


class Feature(ComboFeature):

    def __init__(self, feature):
        super(Feature, self).__init__([feature])
        self.feature = self.features[0]

    def create(self, dataset, train_index=None, force=False):
        self.dataset = dataset
        if self.is_trained():
            self.train_index = train_index
        try:
            if force: raise KeyError
            d = self.dataset.store.load(self.unique_name)
            print "loading '%s' for dataset '%s'" % (self.unique_name,
                self.dataset.name)
            return d
        except KeyError:
            print "creating '%s' for dataset '%s'..." % (self.unique_name,
                self.dataset.name),
            pass
        data = self.feature.create(dataset, train_index, force)
        data = DataFrame(data.copy())
        data = self._create(data)
        data.columns = data.columns.map(self.column_rename)
        self.dataset.store.save(self.unique_name, data)
        if self.is_trained():
            del self.train_index
            del self.train_dataset

        print "done"
        return data

    def _create(self, data):
        return data
# handy shortcut
F = Feature

class MissingIndicator(Feature):
    def _create(self, data):
        for col in data.columns:
            missing = data[col].map(lambda x: int(x.isnan()))
            missing.name = 'missing_%s'%col
            data.append(missing)
        return data

class FillMissing(Feature):
    def __init__(self, feature, fill_value):
        self.fill_value = fill_value
        super(FillMissing, self).__init__(feature)

    def _create(self, data):
        return data.fillna(self.fill_value)

class Length(Feature):
    def _create(self, data):
        return data.applymap(lambda x: len(x) + 1)

class Normalize(Feature):
    def _create(self, data):
        eps = 1.0e-7
        for col in data.columns:
            d = data[col]
            m = d.mean()
            s = d.std()
            if s < eps:
                continue
            data[col] = d.map(lambda x: (x - m)/s)
        return data

class Discretize(Feature):
    def __init__(self, feature, cutoffs, values=None):
        super(Discretize, self).__init__(feature)
        self.cutoffs = cutoffs
        if values is None:
            values = range(cutoffs + 1)
        self.values = values

    def discretize(self, x):
        for i, c in enumerate(self.cutoffs):
            if x < c:
                return self.values[i]
        return self.values[-1]

    def _create(self, data):
        return data.applymap(self.discretize)



class Map(Feature):

    def __init__(self, feature, function, name=None):
        super(Map, self).__init__(feature)
        self.function = function
        if name is None:
            name = function.__name__
        self.name = name
        self._name = name

    def _create(self, data):
        return data.applymap(self.function)


class AsFactor(Feature):

    def _create(self, data):
        factors = set(get_single_column(data))
        # TODO: is this state?
        factors = zip(factors, range(len(factors)))
        mapping = dict(factors)
        self.inverse = dict([(v, k) for k, v in mapping.items()])
        return data.applymap(mapping.get)

    def get_names(self, factor):
        return self.inverse.get(factor)


class AsFactorIndicators(Feature):

    def _create(self, data):
        assert(len(data.columns) == 1)
        col = data.columns[0]
        factors = set(data[col])
        for f in list(factors)[:-1]:
            data['%s-%s'%(f, col)] = data[col].map(lambda x: int(x == f))
        del data[col]
        return data


class IndicatorEquals(Feature):

    def __init__(self, feature, value):
        super(IndicatorEquals, self).__init__(feature)
        self.value = value
        self._name = self._name + '_%s'%value

    def _create(self, data):
        return data.applymap(lambda x: int(x==self.value))


class Log(Map):
    def __init__(self, feature):
        super(Log, self).__init__(feature, math.log)

class Power(Feature):
    def __init__(self, feature, power=2):
        self.power = power
        super(Power, self).__init__(feature)

    def _create(self, data):
        return data.applymap(lambda x: x ** self.power)

class GroupMap(Feature):
    def __init__(self, feature, function, name=None, **groupargs):
        super(GroupMap, self).__init__(feature)
        self.function = function
        if name is None:
            name = function.__name__
        self.name = name
        self._name = name
        self.groupargs = groupargs

    def _create(self, data):
        try:
            d = data.groupby(**self.groupargs).apply(self.function)
            return d
        except ValueError:
            return data.apply(self.function)

class Polynomial(Feature):
    def __init__(self, feature, order=2):
        self.order = order
        super(Polynomial, self).__init__(feature)

    def _create(self, data):
        cols = {}
        for i in range(1, self.order + 1):
            cols[data.name + str(i)] = data ** i
        return DataFrame(cols, index=data.index)

def contain(x, mn, mx):
    if mx is not None and x > mx: return mx
    if mn is not None and x < mn: return mn
    return x

class Contain(Feature):
    def __init__(self, feature, min=None, max=None):
        self.min = min
        self.max = max
        super(Contain, self).__init__(feature)
        self._name = self._name + '(%s,%s)' %(min, max)

    def _create(self, data):
        return data.applymap(lambda x: contain(x, self.min, self.max))


class ReplaceOutliers(Feature):
    def __init__(self, feature, stdevs=7, replace='mean'):
        super(ReplaceOutliers, self).__init__(feature)
        self.stdevs = stdevs
        self.replace = replace
        self._name = self._name + '_%d'%stdevs

    def is_outlier(self, x, mean, std):
        return abs((x-mean)/std) > self.stdevs

    def _create(self, data):
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
    def __init__(self, feature, subset, search=True):
        super(ColumnSubset, self).__init__(feature)
        self.subset = subset
        self.search = search

    def _create(self, data):
        if self.search:
            cols = [c for c in data.columns if any([s in c for s in self.subset])]
        else:
            cols = self.subset
        print cols
        return data[cols]





import combo

