"""
Transforamtions are dataset-wide Features.
They are primarily for convenience and
to keep certain things cleaner (names).
"""
from copy import copy

from ramp.features.base import BaseFeature, ComboFeature


class PreTransformation(object):

    def __init__(self, feature, feature_kwargs):
        self.feature = feature
        self.kwargs = feature_kwargs

    def transform(self, features):
        transformed_features = []
        for feature in features:
            transformed_features.append(inject_feature(feature, self.feature, **self.kwargs))
        return transformed_features


class PostTransformation(object):

    def __init__(self, feature, feature_kwargs, combo=False):
        self.feature = feature
        self.kwargs = feature_kwargs

    def transform(self, features):
        if self.combo:
            return [self.feature(features, **self.kwargs)]
        else:
            return [self.feature(f, **self.kwargs) for f in features]


def inject_feature(feature, feature_to_inject, **kwargs):
    feature = copy(feature)
    if type(feature) == BaseFeature:
        return feature_to_inject(feature, **kwargs)
    if not issubclass(type(feature), ComboFeature):
        return feature
    sub_features = []
    for sub_feature in feature.features:
        if type(sub_feature) == BaseFeature:
            sub_features.append(feature_to_inject(sub_feature, **kwargs))
        else:
            sub_features.append(inject_feature(sub_feature, feature_to_inject, **kwargs))
    feature.set_features(sub_features)
    return feature


def pre_transform_features(features, transform_feature, **kwargs):
    transformed_features = []
    for feature in features:
        transformed_features.append(inject_feature(feature, transform_feature, **kwargs))
    return transformed_features