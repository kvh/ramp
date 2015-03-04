import logging

from pandas import concat, DataFrame, Series, Index
import numpy as np

from ramp.features.base import BaseFeature, Feature, ConstantFeature
from ramp.utils import _pprint, get_single_column


def build_target_safe(target, data, prep_index=None, train_index=None):
    y, ff = target.build(data, prep_index, train_index)
    return get_single_column(y), ff


def apply_target_safe(target, data, fitted_feature):
    y = target.apply(data, fitted_feature)
    return get_single_column(y)


def build_feature_safe(feature, data, prep_index=None, train_index=None):
    d, ff = feature.build(data, prep_index, train_index)

    # sanity check index is valid
    assert d.index.equals(data.index), "%s: %s\n%s" %(feature, d.index, data.index)

    # columns probably shouldn't be constant...
    if not isinstance(feature, ConstantFeature):
        if any(d.std() < 1e-9):
            logging.warn("Feature '%s' has constant column." % feature.unique_name)
    return d, ff


def build_featureset_safe(features, data, prep_index=None, train_index=None):
    # check for dupes
    colnames = set([f.unique_name for f in features])
    assert len(features) == len(colnames), "Duplicate feature: %s" % colnames
    if not features:
        return
    logging.info("Building %d features... " % len(features))
    feature_datas = []
    fitted_features = []
    for feature in features:
        d, ff = build_feature_safe(feature, data, prep_index, train_index)
        feature_datas.append(d)
        fitted_features.append(ff)
    logging.info("Done building features")
    return concat(feature_datas, axis=1), fitted_features


def apply_feature_safe(feature, data, fitted_feature):
    d = feature.apply(data, fitted_feature)

    # sanity check index is valid
    assert d.index.equals(data.index)

    # columns probably shouldn't be constant...
    if not isinstance(feature, ConstantFeature):
        if any(d.std() < 1e-9):
            logging.warn("Feature '%s' has constant column." % feature.unique_name)
    return d


def apply_featureset_safe(features, data, fitted_features):
    assert len(features) == len(fitted_features), "%s, %s" % (len(features),
                                                              len(fitted_features))
    feature_datas = []
    logging.info("Applying %d features to %d data points... " % (len(features), len(data)))
    for f, ff in zip(features, fitted_features):
        feature_datas.append(apply_feature_safe(f, data, ff))
    logging.info("Done applying features")
    return concat(feature_datas, axis=1)


def filter_data_and_indexes(model_def, data, prep_index=None, train_index=None):
    for data_filter in model_def.filters:
        data = data_filter(data)
    if prep_index is not None:
        prep_index = data.index & prep_index
    if train_index is not None:
        train_index = data.index & train_index
    return data, prep_index, train_index

def filter_data(model_def, data):
    return filter_data_and_indexes(model_def, data)[0]


# build_featureset = build_featureset_safe
# build_target = build_target_safe