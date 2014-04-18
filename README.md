Ramp - Rapid Machine Learning Prototyping
=========================================

Ramp is a python library for rapid prototyping of machine learning
solutions. It's a light-weight [pandas](http://pandas.pydata.org)-based 
machine learning framework pluggable with existing 
python machine learning and statistics tools 
([scikit-learn](http://scikit-learn.org), [rpy2](http://rpy.sourceforge.net/rpy2.html), etc.).
Ramp provides a simple, declarative syntax for
exploring features, algorithms and transformations quickly and
efficiently.

Documentation: http://ramp.readthedocs.org

**Why Ramp?**

 *  **Clean, declarative syntax**
    
 *  **Complex feature transformations**

    Chain and combine features:
```python
Normalize(Log('x'))
Interactions([Log('x1'), (F('x2') + F('x3')) / 2])
```
    Reduce feature dimension:
```python
DimensionReduction([F('x%d'%i) for i in range(100)], decomposer=PCA(n_components=3))
```
    Incorporate residuals or predictions to blend with other models:
```python
Residuals(simple_model_def) + Predictions(complex_model_def)
```

 * **Data context awareness**

    Any feature that uses the target ("y") variable will automatically respect the
    current training and test sets. Similarly, preparation data (a feature's mean and stdev, for example)
    is stored and tracked between data contexts.


 *  **Composability**

    All features, estimators, and their fits are composable, pluggable and storable.

 *  **Easy extensibility**

    Ramp has a simple API, allowing you to plug in estimators from
    scikit-learn, rpy2 and elsewhere, or easily build your own feature
    transformations, metrics, feature selectors, reporters, or estimators.


## Quick start
[Getting started with Ramp: Classifying insults](http://www.kenvanharen.com/2012/11/getting-started-with-ramp-detecting.html)

Or, the quintessential Iris example:

```python
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
    metrics=[
        [metrics.GeneralizedMCC()],
        ],
    # report feature importance scores from Random Forest
    reporters=[
        [reporters.RFImportance()],
        ],

    # Try out two algorithms
    model=[
        sklearn.ensemble.RandomForestClassifier(
            n_estimators=20),
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
```

## Status
Ramp is alpha currently, so expect bugs, bug fixes and API changes.

## Requirements
 * Numpy
 * Scipy    
 * Pandas
 * PyTables
 * Sci-kit Learn
 * gensim

## Author
Ken Van Haren. Email with feedback/questions: kvh@science.io [@squaredloss](http://twitter.com/squaredloss)

## Contributors
[John McDonnell](https://github.com/johnmcdonnell)
[Rob Story](https://github.com/wrobstory)
