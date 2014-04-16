  # -*- coding: utf-8 -*-
'''
Store
-------

Data storage classes. The default behavior for this module will always attempt
to read/write from HDF storage first, and will fall back to pickle storage if
required. 

This module uses the MD5 algorithm to create "safe" unique file names based 
on provided key values.  
'''

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
    with open(fname, 'wb') as fout:  # 'b' for binary, needed on Windows
        pickle.dump(obj, fout, protocol=protocol)


def loadpickle(fname):
    """
    Load pickled object from `fname`
    """
    return pickle.load(open(fname, 'rb'))

class Storable(object):
    def save(self, k, v):
        pass
    def load(self, k):
        raise KeyError
    def delete(self, kp):
        pass

class DummyStore(object):
    def save(self, k, v):
        pass
    def load(self, k):
        raise KeyError
    def delete(self, kp):
        pass


class Store(object):

    def __init__(self, path=None, verbose=False):
        """
        ABC for Store classes. Inheriting classes should override get
        and put methods. Currently subclasses for HDF5 and cPickle, but 
        extendable for other data storage types.
        
        Parameters: 
        -----------
        path: string, default None
            Path to data folder
        verbose: bool, default False
            Set 'True' to print read/write messages
        """
        self.path = path
        self._shelf = None
        self._uncachables = set()
        self._cache = {}
        self.verbose = verbose

    def register_uncachable(self, un):
        """Any key containing the substring `un` will NOT be cached """
        self._uncachables.add(un)

    def load(self, key):
        """
        Loads from cache, otherwise defaults to class 'get' method to load
        from store. 
        """
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
        """
        Saves to cache, otherwise defaults to class 'put' method to load
        from store
        """
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
        """Create hex name from key"""
        key_name = re_file.sub('_', key)
        return '_%s__%s' % (hashlib.md5(key).hexdigest()[:10], key_name[:100])

    def get_fname(self, key):
        """Get pickled data path"""
        return os.path.join(self.path, self.safe_name(key))

    def put(self, key, value):
        """Write safe-named data to pickle"""
        dumppickle(value, self.get_fname(key), protocol=0)

    def get(self, key):
        """Load pickled data using key value"""
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
        """HDF store on self.path"""
        return pandas.HDFStore(os.path.join(self.path, 'ramp.h5'))

    def put(self, key, value):
        """Write Pandas DataFrame or Series to HDF store. Other data types
        will default to pickled storage"""
        if isinstance(value, pandas.DataFrame) or isinstance(value, pandas.Series):
            self.get_store()[self.safe_name(key)] = value
        else:
            super(HDFPickleStore, self).put(key, value)

    def get(self, key):
        """Get data from HDF store. If store does not contain key or data, 
        will try to load pickled data."""
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

