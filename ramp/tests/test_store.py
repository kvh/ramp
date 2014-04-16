import sys
sys.path.append('../..')
import tempfile
import unittest

from pandas import DataFrame
from pandas.util.testing import assert_almost_equal

from ramp.result import Result
from ramp.store import DummyStore, HDFPickleStore, PickleStore, MemoryStore


class TestStore(unittest.TestCase):

    def test_dummy(self):
        store = DummyStore()
        key = 'test'
        store.save(key, 1)
        self.assertRaises(KeyError, store.load, (key))
    
    def test_stores(self):
        stores = []
        f = tempfile.mkdtemp()
        stores = [HDFPickleStore(f), PickleStore(f), MemoryStore()]
        
        for store in stores:
            print "Testing store:", store.__class__.__name__
            
            # keys can be arbitrary strings
            key = 'hi there-342 {{ ok }} /<>/'
            store.save(key, 1)
            self.assertEqual(store.load(key), 1)
            
            # test local cache
            self.assertTrue(key in store._cache)
            
            # test python object and overwrite
            store.save(key, dict(hi=1, bye=2))
            self.assertEqual(store.load(key), dict(hi=1, bye=2))
            
            # test pandas object
            store.save(key, DataFrame([range(10), range(10)]))
            assert_almost_equal(store.load(key), DataFrame([range(10), range(10)]))
            
            # test miss
            self.assertRaises(KeyError, store.load, ('not'))
    
#    def test_result_storage(self):
#        print "Testing that result is storable."
#        
#        r = Result(
#                  model_def
#                , y_test = 
#                )

if __name__ == '__main__':
    unittest.main()
