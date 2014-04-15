from configuration import *
from features.base import BaseFeature, Feature, ConstantFeature
from utils import _pprint, get_single_column
from pandas import concat, DataFrame, Series, Index
import numpy as np


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
            print "\nWARNING: Feature '%s' has constant column.\n" % feature.unique_name

    # we probably dont want NANs here...
    #if np.isnan(d.values).any():
        ## TODO HACK: this is not right.  (why isn't it right???)
        #if not feature.unique_name.startswith(
                #Configuration.DEFAULT_PREDICTIONS_NAME):
            #print "\n***** WARNING: NAN in feature '%s' *****\n"%feature.unique_name

    return d


def prep_feature(feature, data_ctx):
    prepped_feature = feature.prepare(data_ctx)
    return prepped_feature


def build_featureset(features, context):
    # check for dupes
    colnames = set([f.unique_name for f in features])
    assert len(features) == len(colnames), "duplicate feature"
    if not features:
        return
    print "Building %d features... " % len(features),
    x = []
    for feature in features:
        x.append(build_feature_safe(feature, context))
    for d in x[1:]:
        assert (d.index == x[0].index).all(), "Mismatched indices after feature creation"
    print "[OK]"
    return concat(x, axis=1)


