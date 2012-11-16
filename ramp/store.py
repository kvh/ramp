import pandas
try:
    import tables
    from tables.exceptions import NoSuchNodeError
except ImportError:
    NoSuchNodeError = None
import cPickle as pickle
#for large objects you have to use pickle due to this bug: http://bugs.python.org/issue13555
#import pickle
import shelve
import hashlib
import os
import re

__all__ = ['DummyStore', 'MemoryStore', 'PickleStore', 'HDFPickleStore']


def dumppickle(obj, fname, protocol=-1):
    """
    Pickle object `obj` to file `fname`.
    """
    with open(fname, 'wb') as fout: # 'b' for binary, needed on Windows
        pickle.dump(obj, fout, protocol=protocol)


def loadpickle(fname):
    """
    Load pickled object from `fname`
    """
    return pickle.load(open(fname, 'rb'))


class DummyStore(object):
    def save(self, k, v): pass
    def load(self, k): raise KeyError
    def delete(self, kp): pass


class Store(object):
    """
    ABC for Store classes. Inheriting classes
    should override get and put methods.
    """
    def __init__(self, path=None, verbose=False):
        self.path = path
        self._shelf = None
        self._uncachables = set()
        self._cache = {}
        self.verbose = verbose

    def register_uncachable(self, un):
        """ any key containing the substring `un` will NOT be cached """
        self._uncachables.add(un)

    def load(self, key):
        try:
            v = self._cache[key]
            if self.verbose:
                print "Retrieving '%s' from local" % key
            return v
        except KeyError:
            v = self.get(key)
            if self.verbose:
                print "Retrieving '%s' from store" % key
            return v

    def save(self, key, value):
        for un in self._uncachables:
            if un in key:
                # print "not caching", key
                return
        self.put(key, value)
        self._cache[key] = value

    def get(self, key):
        raise NotImplementedError

    def put(self, key, value):
        raise NotImplementedError


class MemoryStore(Store):
    """
    Caches values in-memory, no persistence.
    """
    def put(self, key, value): pass
    def get(self, key): raise KeyError


re_file = re.compile(r'\W+')
class PickleStore(Store):
    """
    Pickles values to disk and caches in memory.
    """
    def safe_name(self, key):
        key_name = re_file.sub('_', key)
        return '_%s__%s' % (hashlib.md5(key).hexdigest()[:10], key_name[:100])

    def get_fname(self, key):
        return os.path.join(self.path, self.safe_name(key))

    def put(self, key, value):
        dumppickle(value, self.get_fname(key), protocol=0)

    def get(self, key):
        try:
            return loadpickle(self.get_fname(key))
        except IOError:
            raise KeyError


class HDFPickleStore(PickleStore):
    """
    Attempts to store objects in HDF5 format (numpy/pandas objects). Pickles them
    to disk if that's not possible; also caches values in-memory.
    """
    def get_store(self):
        return pandas.HDFStore(os.path.join(self.path, 'ramp.h5'))

    def put(self, key, value):
        if isinstance(value, pandas.DataFrame) or isinstance(value, pandas.Series):
            self.get_store()[self.safe_name(key)] = value
        else:
            super(HDFPickleStore, self).put(key, value)

    def get(self, key):
        try:
            return self.get_store()[self.safe_name(key)]
        except (KeyError, NoSuchNodeError):
            pass
        return super(HDFPickleStore, self).get(key)


class ShelfStore(Store):
    """
    Deprecated
    """
    def get_store(self):
        if self._shelf is None:
            self._shelf = shelve.open(self.path)
        return self._shelf

    def delete(self, keypart):
        s = self.get_store()
        # TODO: iterating keys is stupid slow for a shelf
        for k in s.keys():
            if keypart in k:
                if self.verbose:
                    print "Deleting '%s' from store"%k
                del s[k]

    def put(self, key, value):
        store = self.get_store()
        store[key] = value
        self._cache[key] = value

    def get(self, key):
        return self.get_store()[key]


try:
    tables
    default_store = HDFPickleStore
except NameError:
    print "Defaulting to basic pickle store. It is recommended \
          you install PyTables for fast HDF5 format."
    default_store = PickleStore

