  # -*- coding: utf-8 -*-
'''
DataContext
-------

The data storage environment of the analysis. 

'''

from hashlib import md5
import copy
from utils import get_np_hashable
from store import HDFPickleStore, MemoryStore, Store, default_store

__all__ = ['DataContext']


class DataContext(object):
    """
    The DataContext is the data storage environment for the Ramp analysis.
    For a given store and pandas index value, Ramp will consider the data
    immutable, and will not check for changes in the data.
    """
    
    def __init__(self, store=None, data=None, train_index=None,
                 prep_index=None, train_once=False):
        """
        Parameters:
        -----------

        store: string or ramp.store.Store object, default None
            An instance of `ramp.store.Store` or a path. If a path, Ramp will
            default to an `HDFPickleStore` at that path if PyTables is
            installed, a `PickleStore` otherwise. Defaults to MemoryStore.
        data: Pandas DataFrame, default None
            Dataframe of data. If all data has been precomputed this
            may not be required.
        train_index: Pandas DataFrame Index, default None
            Pandas Index specifying the data instances to be used in training.
            Stored results will be cached against this.If not provided, the
            entire index of the 'data' parameter will be used.
        prep_index: Pandas DataFrame Index, default None
            Pandas Index specifying the data instances to be
            used in prepping ("x" values). Stored results will be cached
            against this. If not provided, the entire index of `data`
            keyword arg will be used.
        train_once: boolean
            If True, train and predict indexes will not be used as part of key
            hashes, meaning the values from the first run with this context
            will be stored permanently.
        """
        if store is None:
            self.store = MemoryStore()
        else:
            self.store = (store if isinstance(store, Store)
                          else default_store(store))
        self.data = data
        if train_index is not None:
            self.train_index = train_index
        elif self.data is not None:
            self.train_index = self.data.index
        else:
            self.train_index = None

        if prep_index is not None:
            self.prep_index = prep_index
        elif self.data is not None:
            self.prep_index = self.data.index
        else:
            self.prep_index = None
        self.train_once = train_once

    def key_on_indices(self):
        return not self.train_once

    def copy(self):
        """Make a shallow copy of the DataContext"""
        return copy.copy(self)

    def create_key(self):
        """Create hex key using MD5 algorithm."""
        if not self.key_on_indices():
            return ''
        return md5('%s--%s' % (get_np_hashable(self.train_index),
                   get_np_hashable(self.prep_index))).hexdigest()

    def save_context(self, name, config=None):
        """
        Saves this context (specifically it's train and prep indices)
        to it's store with the given name, along with the config, if provided.
        """
        ctx = {'train_index': self.train_index,
               'prep_index': self.prep_index,
               'config': config}
        self.store.save('context__%s' % name, ctx)

    def load_context(self, name):
        """
        Loads a previously saved context with given name,
        assigning the stored training and prep indices
        and returning any stored config.
        """
        ctx = self.store.load('context__%s' % name)
        self.train_index = ctx['train_index']
        self.prep_index = ctx['prep_index']
        return ctx['config']
