Ramp - Rapid Machine Learning Prototyping
========

Ramp is a python module for rapid prototyping of machine learning
solutions. It is essentially a [pandas](http://pandas.pydata.org)
wrapper around various python machine learning and statistics libraries
([scikit-learn](http://scikit-learn.org), [rpy2](http://rpy.sourceforge.net/rpy2.html), etc.),
providing a simple, declarative syntax for
exploring features, algorithms and transformations quickly and
efficiently.

#### Complex feature transformations
Chain basic feature transformations:
    Normalize(Log('x'))
    Interactions([Log('x1'), (F('x2') + F('x3')) / 2])

Reduce feature dimension
    SVDDimensionReduction([F('x%d'%i) for i in range(100)], n_keep=20)

Incorporate residuals or predictions to blend with other models
    Residuals(config_model1) + Predictions(config_model2)

Any feature that uses the target variable will automatically respect the
current training and test sets.

#### Caching
Ramp caches and stores on disk (or elsewhere if you want) all features and models it
computes, so nothing is recomputed unnecessarily. Results are stored and can
be retrieved, compared, blended, and reused between runs.

#### Easily extensible
Ramp has a simple API, allowing you to plug in estimators from
scikit-learn, rpy2 and elsewhere, or build your own feature
transformations, metrics and feature selectors using python's vast
libraries.


### Quick example
    import urllib2
    import tempfile
    import pandas
    import sklearn
    from ramp import *
    from ramp.selectors import RandomForestSelector

    # fetch iris data from UCI
    data = pandas.read_csv(urllib2.urlopen(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"))
    data = data.drop([149])
    columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    data.columns = columns

    # create ramp dataset
    dataset = DataSet(data, name='iris', store=tempfile.mkdtemp() + 'test.store')

    # all features
    features = [FillMissing(f, 0) for f in columns[:-1]]
    # features, log transformed features, and interaction terms
    expanded_features = features + [Log(F(f) + 1) for f in features] + [Interactions(features)]

    # base configuration
    base_conf = Configuration(
        target=AsFactor('class'),
        metric=F1()
        )

    # define several models and feature sets for enumeration
    factory = ConfigFactory(base_conf,
        model=[
            sklearn.ensemble.RandomForestClassifier(n_estimators=20),
            sklearn.linear_model.LogisticRegression(),
            ],
        features=[
            expanded_features,
            # Feature selection
            [FeatureSelector(
                expanded_features,
                RandomForestSelector(classifier=True), # use random forest's importance to trim
                AsFactor('class'), # target to use
                5, # keep top 5 features
                )],
            # Reduce feature dimension (pointless on this dataset)
            [SVDDimensionReduction(expanded_features, n_keep=5)],
            # Normalized features
            [Normalize(f) for f in expanded_features],
            ]
        )

    for conf in factory:
        print conf
        # perform cross validation and report F1 scores
        models.print_scores(models.cv(dataset, conf))

### TODO
- Docs
- More tests
- Simple distributed computation
