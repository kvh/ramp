

class DataContext(object):

    def __init__(self, store, data, train_index=None, prep_index=None):
        self.store = store
        self.data = data
        self.train_index = train_index if train_index is not None else self.data.index
        self.prep_index = prep_index if prep_index is not None else self.data.index
