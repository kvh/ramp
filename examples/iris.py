import pandas
from ramp import *
import urllib2
import sklearn
from sklearn import decomposition


# fetch and clean iris data from UCI
data = pandas.read_csv(urllib2.urlopen(
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


# Define several models and feature sets to explore,
# run 5 fold cross-validation on each and print the results.
# We define 2 models and 4 feature sets, so this will be
# 4 * 2 = 8 models tested.
shortcuts.cv_factory(
    data=data,

    target=[AsFactor('class')],
    metrics=[[metrics.GeneralizedMCC()]],

    # Try out two algorithms
    model=[
        sklearn.ensemble.RandomForestClassifier(n_estimators=20),
        sklearn.linear_model.LogisticRegression(),
        ],

    # and 4 feature sets
    features=[
        expanded_features,

        # Feature selection
        [trained.FeatureSelector(
            expanded_features,
            # use random forest's importance to trim
            selectors.RandomForestSelector(classifier=True),
            target=AsFactor('class'), # target to use
            n_keep=5, # keep top 5 features
            )],

        # Reduce feature dimension (pointless on this dataset)
        [combo.DimensionReduction(expanded_features,
                            decomposer=decomposition.PCA(n_components=4))],

        # Normalized features
        [Normalize(f) for f in expanded_features],
    ]
)
