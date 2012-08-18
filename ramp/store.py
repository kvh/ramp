import pandas
import tables
import cPickle as pickle
import shelve
import hashlib

debug = True

class DummyStore(object):
    def save(self, k, v): pass
    def load(self, k): raise KeyError
    def delete(self, kp): pass

class ShelfStore(object):
    def __init__(self, path):
        self.shelf_path = path
        self._shelf = None
        self._uncachables = set()
        self._cache = {}

    def get_store(self):
        if self._shelf is None:
            self._shelf = shelve.open(self.shelf_path)
        return self._shelf

    def register_uncachable(self, un):
        """ any key containing the substring `un` will NOT be cached """
        self._uncachables.add(un)

    def delete(self, keypart):
        s = self.get_store()
        # TODO: iterating keys is stupid slow for a shelf
        for k in s.keys():
            if keypart in k:
                if debug:
                    print "Deleting '%s' from store"%k
                del s[k]

    def save(self, key, value):
        for un in self._uncachables:
            if un in key:
                # print "not caching", key
                return
        store = self.get_store()
        store[key] = value
        self._cache[key] = value

    def load(self, key):
        try:
            return self._cache[key]
        except KeyError:
            return self.get_store()[key]

