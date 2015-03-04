import sys
sys.path.append('../..')
import unittest

import numpy as np
import pandas as pd
from pandas import DataFrame, Series, Index
from pandas.util.testing import assert_almost_equal

from ramp.builders import *
from ramp.features.base import F, Map
from ramp.metrics import *
from ramp.model_definition import ModelDefinition
from ramp import modeling
from ramp.reporters import *
from ramp.result import Result
from ramp.tests.test_features import make_data


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.data = make_data(100)
        self.result = Result(self.data, self.data,
                             self.data.y, self.data.y,
                             self.data.y, model_def=None,
                             fitted_model=None, original_data=self.data)

    def test_recall(self):
        self.data['y'] = [0]*20 + [1]*80
        self.data['preds'] = [0]*10 + [.5]*60 + [1]*30
        self.result.y_test = self.data.y
        self.result.y_preds = self.data.preds
        m = Recall()
        thresholds = np.arange(0,1,.1)
        expected = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.375, 0.375, 0.375, 0.375]
        assert_almost_equal(expected, [m.score(self.result, t) for t in thresholds])

    def test_weighted_recall(self):
        self.data['y'] = [0]*20 + [1]*80
        self.data['weights'] = [0]*50 + [10]*50
        self.data['preds'] = [0]*10 + [.5]*60 + [.8]*30
        self.result.y_test = self.data.y
        self.result.y_preds = self.data.preds
        self.result.original_data = self.data
        m = WeightedRecall(weight_column='weights')
        thresholds = np.arange(0,1,.1)
        #          [ 0.   0.1  0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]
        expected = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, .6, .6, .6, 0]
        actuals = [m.score(self.result, t) for t in thresholds]
        assert_almost_equal(expected, actuals)

    def test_threshold_metric(self):
        self.data['y'] = [0]*20 + [1]*80
        self.data['preds'] = [0]*10 + [.5]*60 + [1]*30
        self.result.y_test = self.data.y
        self.result.y_preds = self.data.preds
        tm = ThresholdMetric()
        thresholds = [dict(
            threshold=.1,
            expected=dict(
                tp=80,
                tn=10,
                fp=10,
                fn=0
            )),dict(
            threshold=0,
            expected=dict(
                tp=80,
                tn=0,
                fp=20,
                fn=0
            )),dict(
            threshold=1,
            expected=dict(
                tp=30,
                tn=20,
                fp=0,
                fn=50
        ))]
        for d in thresholds:
            threshold = d['threshold']
            expected = d['expected']
            for k, exp_val in expected.items():
                val = getattr(tm, k)(self.result, threshold)
                self.assertEqual(val, exp_val)


class TestMetricReporter(unittest.TestCase):
    def setUp(self):
        self.data = make_data(100)
        self.result = Result(self.data, self.data,
                             self.data.y, self.data.y,
                             self.data.y, model_def=None,
                             fitted_model=None, original_data=self.data)

    def test_metric_reporter(self):
        self.data['y'] = [0]*20 + [1]*80
        self.data['preds'] = [0]*10 + [.5]*60 + [.8]*30
        self.result.y_test = self.data.y
        self.result.y_preds = self.data.preds

        r = MetricReporter(Recall(.7))
        r.process_results([self.result])
        summary = r.summary_df
        n_thresh = 1
        self.assertEqual(len(summary), n_thresh)

        r.plot()

    def test_dual_threshold_reporter(self):
        self.data['y'] = [0]*20 + [1]*80
        self.data['preds'] = [0]*10 + [.5]*60 + [.8]*30
        self.result.y_test = self.data.y
        self.result.y_preds = self.data.preds

        r = DualThresholdMetricReporter(Recall(), PositiveRate())
        r.process_results([self.result])
        summary = r.summary_df
        n_thresh = 3
        self.assertEqual(len(summary), n_thresh)

        r.plot()


if __name__ == '__main__':
    unittest.main()
