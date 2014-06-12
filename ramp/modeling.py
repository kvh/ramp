import pandas as pd

from ramp.builders import (build_featureset_safe,
                           build_target_safe,
                           apply_featureset_safe,
                           apply_target_safe,
                           filter_data_and_indexes,
                           filter_data)
from ramp.estimators.base import FittedEstimator
from ramp.folds import make_default_folds
from ramp.result import Result
from ramp.store import Storable
from ramp.utils import key_from_index


class FittedModel(Storable):

    def __init__(self,
                 model_def,
                 fitted_features,
                 fitted_target,
                 fitted_estimator):
        self.model_def = model_def
        self.fitted_features = fitted_features
        self.fitted_target = fitted_target
        self.fitted_estimator = fitted_estimator


class PackagedModel(Storable):

    def __init__(self, fitted_model, data, data_description, result=None, reports=None):
        super(PackagedModel, self).__init__()
        self.fitted_model = fitted_model
        self.n_rows = len(data)
        self.n_cols = len(data.columns)
        self.data_key = key_from_index(data.index)
        self.data_description = data_description
        self.result = result
        self.reports = reports


def generate_train(model_def, data, prep_index=None, train_index=None):
    # create training set
    data, prep_index, train_index = filter_data_and_indexes(model_def, data, prep_index, train_index)
    x_train, fitted_features = build_featureset_safe(model_def.features, data, prep_index, train_index)
    y_train, fitted_target = build_target_safe(model_def.target, data, prep_index, train_index)
    return x_train, y_train, fitted_features, fitted_target


def generate_test(model_def, predict_data, fitted_model, compute_actuals=True):
    # create test set
    predict_data = filter_data(model_def, predict_data)
    x_test = apply_featureset_safe(model_def.features, predict_data, fitted_model.fitted_features)
    if compute_actuals:
        y_test = apply_target_safe(model_def.target, predict_data, fitted_model.fitted_target)
    else:
        y_test = None
    return x_test, y_test
# ughh, this is so nose doesn't pick this up as a test
generate_test.__test__ = False


def fit_model(model_def, data, prep_index=None, train_index=None):
    x_train, y_train, fitted_features, fitted_target = generate_train(model_def,
                                                                      data,
                                                                      prep_index,
                                                                      train_index)

    # fit estimator
    model_def.estimator.fit(x_train, y_train)

    fitted_estimator = FittedEstimator(model_def.estimator, x_train, y_train)

    fitted_model = FittedModel(model_def, fitted_features, fitted_target, fitted_estimator)
    return x_train, y_train, fitted_model


def build_fitted_model(*args, **kwargs):
    x, y, fitted_model = fit_model(*args, **kwargs)
    return fitted_model


def predict_with_model(model_def, data, fitted_model, compute_actuals=False):
    x_test, y_test = generate_test(model_def, data, fitted_model, compute_actuals)
    y_preds = fitted_model.fitted_estimator.predict(x_test)
    return pd.Series(y_preds, index=x_test.index)


def fit_and_predict(model_def, data, prep_index=None, train_index=None):
    x, y, fitted_model = fit_model(model_def, data, prep_index, train_index)
    return predict_with_model(model_def, data, fitted_model)


def cross_validate(model_def, data, folds, reporters=[], repeat=1):
    """
    """
    results = []

    if isinstance(folds, int):
        folds = make_default_folds(num_folds=folds, data=data)

    for i in range(repeat):
        for fold in folds:
            if len(fold) == 2:
                train_index, test_index = fold
                prep_index = None
            elif len(fold) == 3:
                train_index, test_index, prep_index = fold
            else:
                raise ValueError("Fold is not of right dimension (%d, not 2 or 3)"%len(fold))
            x_train, y_train, fitted_model = fit_model(model_def, data, prep_index, train_index)
            x_test, y_test = generate_test(model_def, data, fitted_model)
            y_preds = fitted_model.fitted_estimator.predict(x_test)
            result = Result(x_train, x_test, y_train, y_test, y_preds, model_def, fitted_model, data)
            results.append(result)

            for reporter in reporters:
                reporter.update(result)
    return results, reporters


def build_and_package_model(model_def, data, data_description=None, evaluate=False,
                            reporters=None, prep_index=None, train_index=None):
    x_train, y_train, fitted_model = fit_model(model_def, data, prep_index, train_index)
    y_preds = fitted_model.fitted_estimator.predict(x_train)
    result = None
    if evaluate:
        # only evaluate on train (this seems reasonable)
        result = Result(x_train, x_train, y_train, y_train, y_preds, model_def, fitted_model, data)
        #TODO
        # reports = evaluate(result, reporters)

    # TODO
    reports = []

    packaged_model = PackagedModel(fitted_model,
                                   data,
                                   data_description,
                                   result,
                                   reports)
    return packaged_model
