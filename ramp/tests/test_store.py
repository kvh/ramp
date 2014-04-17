import sys
sys.path.append('../..')
import tempfile
import unittest

import pandas as pd
from pandas import DataFrame
from pandas.util.testing import assert_almost_equal

from ramp.features import F, FittedFeature
from ramp.result import Result
from ramp.store import *


class TestStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_storable(self):
        f = FittedFeature(F('a'), pd.Index([]), pd.Index([]))
        f.to_pickle(self.tmp + 'tst')

        f2 = FittedFeature.from_pickle(self.tmp + 'tst')
        self.assertEqual(f2.prep_n, f.prep_n)


if __name__ == '__main__':
    unittest.main()
