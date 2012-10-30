from core import get_dataset, _register_dataset
from store import ShelfStore, DummyStore, PickleStore, Store
from configuration import *
import core
from features.base import BaseFeature, Feature, ConstantFeature
from pandas import concat, DataFrame, Series, Index
import hashlib
from sklearn import cross_validation
from sklearn import feature_selection
import scipy
import numpy as np
import re
import os
import random
import copy
from utils import _pprint, get_single_column


def build_target(target, context):
    y = target.create(context)
    return get_single_column(y)


def build_feature_safe(feature, context):
    d = feature.create(context)

    # sanity check index is valid
    assert not d.index - context.data.index

    # columns probably shouldn't be constant...
    if not isinstance(feature, ConstantFeature):
        if any(d.std() < 1e-9):
            print "\n\nWARNING: Feature '%s' has constant column. \n\n" % feature.unique_name

    # we probably dont want NANs here...
    if np.isnan(d.values).any():
        # TODO HACK: this is not right.  (why isn't it right???)
        if not feature.unique_name.startswith(
                Configuration.DEFAULT_PREDICTIONS_NAME):
            print "\n\n***** WARNING: NAN in feature '%s' *****\n\n"%feature.unique_name

    return d


def build_featureset(features, context):
    # check for dupes
    colnames = set([f.unique_name for f in features])
    assert len(features) == len(colnames), "duplicate feature"
    if not features:
        return
    x = []
    for feature in features:
        x.append(build_feature_safe(feature, context))
    for d in x[1:]:
        assert (d.index == x[0].index).all(), "Mismatched indices after feature creation"
    return concat(x, axis=1)

