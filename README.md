Ramp - Rapid Machine Learning Prototyping
========

Ramp is a python module for rapid prototyping of machine learning
solutions. It is essentially a [pandas](http://pandas.pydata.org)
wrapper around various python machine learning and statistics libraries
([scikit-learn](http://scikit-learn.org), [rpy2](http://rpy.sourceforge.net/rpy2.html), etc.),
providing a simple, declarative syntax for
exploring features, algorithms and transformations quickly and
efficiently.

Documentation: http://ramp.readthedocs.org

## Complex feature transformations
Chain and combine features:

    Normalize(Log('x'))
    Interactions([Log('x1'), (F('x2') + F('x3')) / 2])

Reduce feature dimension:

    SVDDimensionReduction([F('x%d'%i) for i in range(100)], n_keep=20)

Incorporate residuals or predictions to blend with other models:

    Residuals(config_model1) + Predictions(config_model2)
Any feature that uses the target ("y") variable will automatically respect the
current training and test sets.

## Caching
Ramp caches and stores on disk in fast HDF5 format (or elsewhere if you want) all features and models it
computes, so nothing is recomputed unnecessarily. Results are stored 
and can be retrieved, compared, blended, and reused between runs.

## Easy extensibility
Ramp has a simple API, allowing you to plug in estimators from
scikit-learn, rpy2 and elsewhere, or easily build your own feature
transformations, metrics, feature selectors, reporters, or estimators.


## Quick example
[Getting started with Ramp: Classifying insults](http://kenvanharen.com/)

## Status
Ramp is very alpha currently, so expect bugs, bug fixes and API changes.

## Requirements
 * Numpy
 * Scipy    
 * Pandas
 * PyTables
 * Sci-kit Learn
