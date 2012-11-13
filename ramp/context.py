from hashlib import md5
import copy
from utils import get_np_hashable
from store import HDFPickleStore, Store, default_store

__all__ = ['DataContext']


class DataContext(object):

    def __init__(self, store, data=None, train_index=None, prep_index=None):
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
        ctx = {'train_index':self.train_index,
                'prep_index':self.prep_index,
                'config':config}
        self.store.save('context__%s' % name, ctx)

    def load_context(self, name):
        ctx = self.store.load('context__%s' % name)
        self.train_index = ctx['train_index']
        self.prep_index = ctx['prep_index']
        return ctx['config']
