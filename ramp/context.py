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
        """
        if store is None:
            self.store = MemoryStore()
        else:
            self.store = (store if isinstance(store, Store)
                          else default_store(store))
        self.data = data

