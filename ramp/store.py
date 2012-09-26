import pandas
import tables
import cPickle
# have to use pickle because of this bug: http://bugs.python.org/issue13555
import pickle
import shelve
import hashlib
import os
import re

debug = True


def dumppickle(obj, fname, protocol=-1):
    """Pickle object `obj` to file `fname`."""
    with open(fname, 'wb') as fout: # 'b' for binary, needed on Windows
        pickle.dump(obj, fout, protocol=protocol)


def loadpickle(fname):
    """Load pickled object from `fname`"""
    return pickle.load(open(fname, 'rb'))


class DummyStore(object):
    def save(self, k, v): pass
    def load(self, k): raise KeyError
    def delete(self, kp): pass


class Store(object):
    def __init__(self, path):
        self.path = path
        self._shelf = None
        self._uncachables = set()
        self._cache = {}

    def register_uncachable(self, un):
        """ any key containing the substring `un` will NOT be cached """
        self._uncachables.add(un)

    def load(self, key):
        try:
            return self._cache[key]
        except KeyError:
            return self.get(key)

    def save(self, key, value):
        for un in self._uncachables:
            if un in key:
                # print "not caching", key
                return
        self.put(key, value)
        self._cache[key] = value


re_file = re.compile(r'\W+')
class PickleStore(Store):

    def get_fname(self, key):
        key_name = re_file.sub('_', key)
        return os.path.join(self.path, hashlib.md5(key).hexdigest()[:10] + '--' + key_name)

    def put(self, key, value):
        dumppickle(value, self.get_fname(key), protocol=0)

    def get(self, key):
        try:
            return loadpickle(self.get_fname(key))
        except IOError:
            raise KeyError


class ShelfStore(Store):

    def get_store(self):
        if self._shelf is None:
            self._shelf = shelve.open(self.path)
        return self._shelf

    def delete(self, keypart):
        s = self.get_store()
        # TODO: iterating keys is stupid slow for a shelf
        for k in s.keys():
            if keypart in k:
                if debug:
                    print "Deleting '%s' from store"%k
                del s[k]

    def put(self, key, value):
        store = self.get_store()
        store[key] = value
        self._cache[key] = value

    def get(self, key):
        return self.get_store()[key]

