  # -*- coding: utf-8 -*-
'''
ModelDefinition
-------

A configuration is a uniquely defined data analysis model, including
features, estimator, and target metric. ModelDefinitions can be pickled
and retrieved.

The ConfigFactory is at the core of the power of Ramp. It creates a
configuration iterator that allows for the exploration of a large number
of features, models, and metrics

'''
import copy
import itertools
import logging

from ramp.features.base import BaseFeature, Feature, AsFactorIndicators, FillMissing
from ramp.estimators.base import wrap_sklearn_like_estimator
from ramp.filters import filter_incomplete
from ramp.transformations import pre_transform_features
from ramp.utils import _pprint, stable_repr

__all__ = ['ModelDefinition', 'model_definition_factory']


class ModelDefinition(object):
    """
    Defines a specific data analysis model,
    including features, estimator and target metric.
    Can be stored (pickled) and retrieved.
    """
    DEFAULT_PREDICTIONS_NAME = '$predictions'
    params = ['target', 'features', 'estimator', 'column_subset'
              'prediction', 'predictions_name', 'actual']

    def __init__(self,
                 target=None,
                 features=None,
                 estimator=None,
                 prediction=None,
                 predictions_name=None,
                 actual=None,
                 column_subset=None,
                 filters=None,
                 fill_missing=None,
                 discard_incomplete=False,
                 categorical_indicators=False,
                 weight=None):
        """
        Parameters:
        ___________

        target: `Feature` or string, default None
            `Feature` or basestring specifying the target ("y") variable of
            the analysis.

        features: `Feature`, default None
            An iterable of `Features <Feature>` to be used by the estimator
            in the analysis.

        estimator: estimator (compatible with sklearn estimators), default None
            An estimator instance compatible with sklearn estimator
            conventions: Has fit(x, y) and predict(y) methods. If the object is
            not a ramp Estimator, it will be wrapped to add sensible
            prediction methods.

        predictions_name: string, default None
            A unique string used as a column identifier for model predictions.
            Must be unique among all feature names: eg '$logreg_predictions$'

        prediction: `Feature`, default None
            A `Feature` transformation of the special `predictions_name`
            column used to post-process predictions prior to metric scoring.

        actual: `Feature`, default None
            `Feature`. Used if `target` represents a transformation that is
            NOT the actual target "y" values. Used in conjuction with
            `prediction` to allow model training, predictions and scoring to
            operate on different values.
        """
        self.set_attrs(target,
                       features,
                       estimator,
                       prediction,
                       predictions_name,
                       actual,
                       column_subset,
                       filters,
                       fill_missing,
                       discard_incomplete,
                       categorical_indicators,
                       weight)


    def set_attrs(self,
                  target=None,
                  features=None,
                  estimator=None,
                  prediction=None,
                  predictions_name=None,
                  actual=None,
                  column_subset=None,
                  filters=None,
                  fill_missing=None,
                  discard_incomplete=False,
                  categorical_indicators=False,
                  weight=None):

        if prediction is not None:
            if predictions_name is None:
                raise ValueError("If you provide a prediction feature, you "
                "must also specify a _unique_ 'predictions_name'")

        if isinstance(target, BaseFeature) or target is None:
            self.target = target
        else:
            self.target = Feature(target)

        if isinstance(prediction, BaseFeature) or prediction is None:
            self.prediction = prediction
        else:
            self.prediction = Feature(prediction)
        self.predictions_name = predictions_name

        if actual is None:
            actual = self.target
        self.actual = (actual if isinstance(actual, BaseFeature)
                       else Feature(actual))

        self.filters = filters if filters else []
        if discard_incomplete:
            self.filters.append(filter_incomplete)

        if features:
            self.features = ([f if isinstance(f, BaseFeature) else Feature(f)
                              for f in features])

            if categorical_indicators:
                self.features = pre_transform_features(self.features,
                                                       AsFactorIndicators,
                                                       only_if_categorical=True)
            if fill_missing is not None:
                self.features = pre_transform_features(self.features,
                                                       FillMissing,
                                                       fill_value=missing)
        else:
            self.features = None

        self.weight = weight

        # Wrap estimator to return probabilities in the case of a classifier
        self.estimator = wrap_sklearn_like_estimator(estimator)

        self.column_subset = column_subset

    def __getstate__(self):
        # shallow copy dict and keep references
        dct = self.__dict__.copy()
        return dct

    def __repr__(self):
        return stable_repr(self)

    def __str__(self):
        if self.features is not None:
            feature_count = len(self.features)
        else:
            feature_count = 0
        return 'estimator: %s\nfeatures: %d [%s ...]\ntarget: %s' % (
            self.estimator,
            feature_count,
            ' '.join([str(f) for f in self.features])[:50],
            self.target
        )

    @property
    def summary(self):
        """
        Summary of model definition for labeling. Intended to be somewhat
        readable but unique to a given model definition.
        """
        if self.features is not None:
            feature_count = len(self.features)
        else:
            feature_count = 0
        feature_hash = 'feathash:' + str(hash(tuple(self.features)))
        return (str(self.estimator), feature_count, feature_hash, self.target)

    def update(self, dct):
        """Update the configuration with new parameters. Must use same
        kwargs as __init__"""
        d = self.__dict__.copy()
        d.update(dct)
        self.set_attrs(**d)


def model_definition_factory(model_definition, **kwargs):
    """
    Provides an iterator over passed-in
    configuration values, allowing for easy
    exploration of models.

    Parameters:
    ___________

    base_config:
        The base `ModelDefinition` to augment

    kwargs:
        Can be any keyword accepted by `ModelDefinition`.
        Values should be iterables.
    """
    if not kwargs:
        yield config
    else:
        for param in kwargs:
            if not hasattr(model_definition, param):
                raise ValueError("'%s' is not a valid configuration parameter" % param)

        for raw_params in itertools.product(*kwargs.values()):
            new_definition = copy.copy(model_definition)
            new_definition.update(dict(zip(kwargs.keys(), raw_params)))
            yield new_definition

