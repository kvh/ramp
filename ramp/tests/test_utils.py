import sys
sys.path.append('../..')
from ramp.utils import *
from ramp.features.base import *
import unittest
from pandas import *
import tempfile


class TestUtils(unittest.TestCase):

    def test_np_hashes(self):
        a = np.random.randn(20)
        h = get_np_hash(a)
        a[0] = 200000
        h2 = get_np_hash(a)
        self.assertNotEqual(h, h2)
        b = a[0]
        a[0] = b
        self.assertEqual(h2, get_np_hash(a))

    def test_stable_repr(self):
        f = F('test')
        f2 = F('test')
        # equalivalent objects should have same repr
        self.assertEqual(stable_repr(f), stable_repr(f2))

        # repr is independent of object ids
        class Test: pass
        f.f = Test()
        r1 = stable_repr(f)
        f.f = Test()
        r2 = stable_repr(f)
        self.assertEqual(r1, r2)


if __name__ == '__main__':
    unittest.main(verbosity=2)

