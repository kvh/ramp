import urllib2

import pandas as pd
import sklearn
from sklearn import decomposition

import ramp
from ramp.features import *
from ramp.metrics import PositiveRate, Recall


# fetch and clean iris data from UCI
data = pd.read_csv(urllib2.urlopen(
    "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"))
data = data.drop([149]) # bad line
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
data.columns = columns


# all features
features = [FillMissing(f, 0) for f in columns[:-1]]

# features, log transformed features, and interaction terms
expanded_features = (
    features +
    [Log(F(f) + 1) for f in features] +
    [
        F('sepal_width') ** 2,
        combo.Interactions(features),
    ]
)

reporters = [
    ramp.reporters.MetricReporter(Recall(.4)),
    ramp.reporters.DualThresholdMetricReporter(Recall(), PositiveRate())
]


# Define several models and feature sets to explore,
# run 5 fold cross-validation on each and print the results.
# We define 2 models and 4 feature sets, so this will be
# 4 * 2 = 8 models tested.
ramp.shortcuts.cv_factory(
    data=data,
    folds=5,

    target=[AsFactor('class')],

    reporters=reporters,

    # Try out two algorithms
    estimator=[
        sklearn.ensemble.RandomForestClassifier(
            n_estimators=20),
        sklearn.linear_model.LogisticRegression(),
        ],

    # and 4 feature sets
    features=[
        expanded_features,

        # Feature selection
        # [trained.FeatureSelector(
        #     expanded_features,
        #     # use random forest's importance to trim
        #     ramp.selectors.BinaryFeatureSelector(),
        #     target=AsFactor('class'), # target to use
        #     data=data,
        #     n_keep=5, # keep top 5 features
        #     )],

        # Reduce feature dimension (pointless on this dataset)
        [combo.DimensionReduction(expanded_features,
                            decomposer=decomposition.PCA(n_components=4))],

        # Normalized features
        [Normalize(f) for f in expanded_features],
    ]
)

for reporter in reporters:
    reporter.plot()
