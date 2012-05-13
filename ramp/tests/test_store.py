import sys
sys.path.append('../..')
from ramp.features import *
from ramp.dataset import *
from ramp.models import *
from ramp.metrics import *
from ramp.store import DummyStore, ShelfStore
import unittest
import pandas
import tempfile
import shelve


class TestShelfStore(unittest.TestCase):

    def test_dummy(self):
        store = DummyStore()
        store.save('test', 1)
        self.assertRaises(KeyError, store.load, ('test'))

    def test_store(self):
        p = tempfile.mkdtemp()
        store = ShelfStore(p + 'test.shelf')
        k = 'test'
        v = 123
        store.save(k, v)
        self.assertEqual(store.load(k), v)
        store._shelf.close()
        # re-open manually
        shelf = shelve.open(p + 'test.shelf')
        self.assertEqual(len(shelf.keys()), 1)
        self.assertEqual(shelf[k], v)



