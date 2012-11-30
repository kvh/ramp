from pandas import Series, DataFrame, concat
import numpy as np
import random
import inspect
import math
import re
from hashlib import md5
from ..utils import _pprint, get_np_hashable, get_single_column, stable_repr


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

    def depends_on_y(self):
        return False

    def depends_on_other_x(self):
        return False

    def create(self, context, *args, **kwargs):
        return DataFrame(context.data[self.feature],
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

    def create(self, context, *args, **kwargs):
        return DataFrame(
                [self.feature] * len(context.data),
                index=context.data.index,
                columns=['%s' % self.feature])


class DummyFeature(BaseFeature):
    """ For testing """

    def __init__(self):
        self.feature = ''

    def create(self, context, *args, **kwargs):
        return context.data


class ComboFeature(BaseFeature):
    """
    Abstract base for more complex features
    """

    hash_length = 8
    _cacheable = True
    re_hsh = re.compile(r' \[\w{%d}\]' % hash_length)

    def __init__(self, features):
        """
        Inheriting classes responsible for setting human-readable description of
        feature and parameters on _name attribute.
        """
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
        # shallow copy dict and keep references
        dct = self.__dict__.copy()
        # HACK remove temporary state
        if 'context' in dct:
            del dct['context']
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
        A readable version of this feature (and its contained features)
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
        return any([f.depends_on_y() for f in self.features])

    def depends_on_other_x(self):
        if hasattr(self, '_prepare'):
            return True
        return any([f.depends_on_other_x() for f in self.features])

    def create_data(self, force):
        datas = []
        # recurse
        for feature in self.features:
            data = feature.create(self.context, force)
            # copy the dataframe to isolate side effects
            # TODO: is this really necessary? Can we enforce immutability?
            #data = DataFrame(data.copy())
            datas.append(data)
        # actually apply the feature
        data = self._create(datas)
        return data

    def get_prep_key(self):
        """
        Stable, unique key for this feature and a given prep_index and train_index.
        we key on train_index as well because prep data may involve training.
        """
        s = get_np_hashable(self.context.prep_index)
        tindex = get_np_hashable(self.context.train_index) if self.depends_on_y() else ''
        return self.unique_name + '--prep--' + md5('%s--%s' % (s, tindex)).hexdigest()

    def get_prep_data(self, data=None, force=False):
        try:
            if force: raise KeyError
            d = self.context.store.load(self.get_prep_key())
            return d
        except KeyError:
            if data is None:
                raise KeyError()
        print "Prepping '%s'" % self.unique_name
        prep_data = self._prepare(data.reindex(self.context.prep_index))
        self.context.store.save(self.get_prep_key(), prep_data)
        return prep_data

    def create_key(self):
        s = get_np_hashable(self.context.data.index)
        tindex = get_np_hashable(self.context.train_index) if self.depends_on_y() else ''
        pindex = get_np_hashable(self.context.prep_index) if self.depends_on_other_x() else ''
        return self.unique_name + '--' + md5('%s--%s--%s' % (s, tindex, pindex)).hexdigest()

    def create(self, context, force=False):
        """ Caching wrapper around actual feature creation """

        # save existing context
        prev_context = getattr(self, "context", None)

        self.context = context

        try:
            if force: raise KeyError
            d = self.context.store.load(self.create_key())
            print "loading '%s'" % (self.unique_name)
            #TODO: repeated... use 'with' maybe?
            del self.context
            return d
        except KeyError:
            print "creating '%s' ..." % (self.unique_name)

        data = self.create_data(force)

        # cache it
        if self._cacheable:
            self.context.store.save(self.create_key(), data)

        # reassign previous context (typically None)
        # this is for edge case of same feature object nested in itself
        self.context = prev_context

        return data

    def _create(self, datas):
        """
        Actual feature creation.
        """
        data = self.combine(datas)
        hsh = self._hash() # cache this so we dont recompute for every column
        data.columns = data.columns.map(lambda x: self.column_rename(x, hsh))
        return data

    def combine(self, datas):
        """
        Needs to be overridden
        """
        raise NotImplementedError


class Feature(ComboFeature):
    """
    Base class for features operating on a single feature.
    """
    def __init__(self, feature):
        super(Feature, self).__init__([feature])
        self.feature = self.features[0]

    def create_data(self, force):
        """
        Overrides `ComboFeature` create_data method to only
        operate on a single sub-feature.
        """
        data = self.feature.create(self.context, force)
        #data = DataFrame(data.copy())
        data = self._create(data)
        hsh = self._hash() # cache this so we dont recompute for every column
        data.columns = data.columns.map(lambda x: self.column_rename(x, hsh))
        return data

    def _create(self, data):
        """
        Should be overriden by inheriting classes.
        """
        return data
# handy shortcut
F = Feature


class MissingIndicator(Feature):
    """
    Adds a missing indicator column for this feature.
    Indicator will be 1 if given feature `isnan` (numpy definition), 0 otherwise.
    """
    def _create(self, data):
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

    def _create(self, data):
        return data.fillna(self.fill_value)


class Length(Feature):
    """
    Applies builtin `len` to feature.
    """
    def _create(self, data):
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

    def _create(self, data):
        eps = 1.0e-7
        col_stats = self.get_prep_data(data)
        d = DataFrame(index=data.index)
        for col in data.columns:
            m, s = col_stats.get(col, (0, 0))
            if s < eps:
                continue
            d[col] = data[col].map(lambda x: (x - m)/s)
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

    def _create(self, data):
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

    def _create(self, data):
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

    def _create(self, data):
        levels = self.get_prep_data(data)
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
    would be mapped to two indicator columns (the
    third implied by zeros on the other two columns)
    """
    def __init__(self, feature, levels=None):
        super(AsFactorIndicators, self).__init__(feature)
        self.levels = levels

    def _prepare(self, data):
        levels = self.levels
        if not levels:
            levels = sorted(set(get_single_column(data)))
        return levels

    def _create(self, data):
        factors = self.get_prep_data(data)
        data = get_single_column(data)
        d = DataFrame(index=data.index)
        for f in list(factors)[:-1]:
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

    def _create(self, data):
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

    def _create(self, data):
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

    def _create(self, data):
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
        prep = self.get_prep_data(data)
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

    def _create(self, data):
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
    def __init__(self, feature, subset, match_substr=False):
        super(ColumnSubset, self).__init__(feature)
        self.subset = subset
        self.match_substr = match_substr

    def _create(self, data):
        if self.match_substr:
            cols = [c for c in data.columns if any([s in c for s in self.subset])]
        else:
            cols = self.subset
        return data[cols]


import combo

