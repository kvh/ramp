from hashlib import md5
import copy
from utils import get_np_hashable


class DataContext(object):

    def __init__(self, store, data, train_index=None, prep_index=None):
        self.store = store
        self.data = data
        self.train_index = train_index if train_index is not None else self.data.index
        self.prep_index = prep_index if prep_index is not None else self.data.index

    def copy(self):
        return copy.copy(self)

    def create_key(self):
        return md5('%s--%s' % (get_np_hashable(self.train_index),
            get_np_hashable(self.prep_index))).hexdigest()
