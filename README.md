Ramp - Rapid Machine Learning Prototyping
=========================================

Ramp is a python module for rapid prototyping of machine learning
solutions. It is essentially a [pandas](http://pandas.pydata.org)
wrapper around various python machine learning and statistics libraries
([scikit-learn](http://scikit-learn.org), [rpy2](http://rpy.sourceforge.net/rpy2.html), etc.),
providing a simple, declarative syntax for
exploring features, algorithms and transformations quickly and
efficiently.

Documentation: http://ramp.readthedocs.org

**Why Ramp?**

 *  **Clean, declarative syntax**
    
    No more hackish one-off spaghetti scripts!

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
Residuals(config_model1) + Predictions(config_model2)
```
    Any feature that uses the target ("y") variable will automatically respect the
    current training and test sets.


 *  **Caching**

    Ramp caches and stores on disk in fast HDF5 format (or elsewhere if you want) all features and models it
    computes, so nothing is recomputed unnecessarily. Results are stored 
    and can be retrieved, compared, blended, and reused between runs.

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
```

## Status
Ramp is very alpha currently, so expect bugs, bug fixes and API changes.

## Requirements
 * Numpy
 * Scipy    
 * Pandas
 * PyTables
 * Sci-kit Learn

## Author
Ken Van Haren. Email with feedback/questions: kvh@science.io
