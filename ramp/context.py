from hashlib import md5
import copy
from utils import get_np_hashable
from store import HDFPickleStore, MemoryStore, Store, default_store

__all__ = ['DataContext']


class DataContext(object):
    """
    All Ramp analyses require a DataContext.
    A DataContext represents the environment of the
    analysis. Most importantly this means for a given store
    and pandas index value, Ramp will consider the data immutable --
    it will not check the data again to see if it has changed.
    """
    def __init__(self, store=None, data=None, train_index=None, prep_index=None):
        """
        **Args**

        store: An instance of `store.Store` or a path. If a path
                Ramp will default to an `HDFPickleStore` at that path
                if PyTables is installed, a `PickleStore` otherwise.
                Defaults to MemoryStore.
        data: a pandas DataFrame. If all data has been precomputed this
                may not be required.
        train_index: a pandas Index specifying the data instances to be
                used in training. Stored results will be cached against this.
                If not provided, the entire index of `data` will be used.
        prep_index: a pandas Index specifying the data instances to be
                used in prepping ("x" values). Stored results will be cached against this.
                If not provided, the entire index of `data` will be used.
        """
        if store is None:
            self.store = MemoryStore()
        else:
            self.store = store if isinstance(store, Store) else default_store(store)
        self.data = data
        self.train_index = train_index if train_index is not None else self.data.index if self.data is not None else None
        self.prep_index = prep_index if prep_index is not None else self.data.index if self.data is not None else None

    def copy(self):
        return copy.copy(self)

    def create_key(self):
        return md5('%s--%s' % (get_np_hashable(self.train_index),
            get_np_hashable(self.prep_index))).hexdigest()

    def save_context(self, name, config=None):
        """
        Saves this context (specifically it's train and prep indices)
        to it's store with the given
        name, along with the config, if provided.
        """
        ctx = {'train_index':self.train_index,
                'prep_index':self.prep_index,
                'config':config}
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
