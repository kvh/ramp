import pandas as pd

from ramp.builders import (build_featureset_safe,
                           build_target_safe,
                           apply_featureset_safe,
                           apply_target_safe,
                           filter_data_and_indexes,
                           filter_data)
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

    def __init__(self, fitted_model, data, data_description=None,
                 built_at=None, result=None, reports=None):
        super(PackagedModel, self).__init__()
        self.fitted_model = fitted_model
        self.n_rows = len(data)
        self.n_cols = len(data.columns)
        self.data_key = key_from_index(data.index)
        self.data_description = data_description
        self.built_at = built_at
        self.result = result
        self.reports = reports


def generate_train(model_def, data, prep_index=None, train_index=None):
    # create training set
    data, prep_index, train_index = filter_data_and_indexes(model_def, data, prep_index, train_index)
    x_train, fitted_features = build_featureset_safe(model_def.features, data, prep_index, train_index)
    y_train, fitted_target = build_target_safe(model_def.target, data, prep_index, train_index)
    x_train = x_train.reindex(train_index)
    y_train = y_train.reindex(train_index)
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


def fit_model(model_def, data, prep_index=None, train_index=None, **fit_args):
    #TODO: There should be an option for computing y_train_predictions
    x_train, y_train, fitted_features, fitted_target = generate_train(model_def,
                                                                      data,
                                                                      prep_index,
                                                                      train_index)

    # fit estimator
    if fit_args:
        model_def.estimator.fit(x_train, y_train, **fit_args)
    else:
        model_def.estimator.fit(x_train, y_train)

    fitted_estimator = model_def.estimator

    fitted_model = FittedModel(model_def, fitted_features, fitted_target, fitted_estimator)
    return x_train, y_train, fitted_model


def build_fitted_model(*args, **kwargs):
    x, y, fitted_model = fit_model(*args, **kwargs)
    return fitted_model


def predict(fitted_model, x_data):
    model_def = fitted_model.model_def
    predictions = fitted_model.fitted_estimator.predict(x_data)
    predictions = pd.Series(predictions, index=x_data.index)
    if model_def.evaluation_transformation is not None:
        x_data[model_def.predictions_name] = predictions
        predictions, ff = build_target_safe(model_def.evaluation_transformation, x_data)
        del x_data[model_def.predictions_name]
    return predictions


def predict_with_model(fitted_model, data, compute_actuals=False):
    model_def = fitted_model.model_def
    x_test, y_test = generate_test(model_def, data, fitted_model, compute_actuals)
    return predict(fitted_model, x_test)


def fit_and_predict(model_def, data, prep_index=None, train_index=None):
    x, y, fitted_model = fit_model(model_def, data, prep_index, train_index)
    return predict_with_model(model_def, data, fitted_model)


def cross_validate(model_def, data, folds, repeat=1):
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
            assert len(train_index & test_index) == 0, "train and test overlap!!! %s, %s" % (train_index, test_index)
            x_train, y_train, fitted_model = fit_model(model_def, data, prep_index, train_index)
            test_data = data.loc[test_index]
            x_test, y_test = generate_test(model_def, test_data, fitted_model)
            assert len(x_train.index & x_test.index) == 0, "train and test overlap!!! %s" % (x_train.index & x_test.index)
            y_preds = predict(fitted_model, x_test)
            if model_def.evaluation_target is not None:
                y_test, ff = build_target_safe(model_def.evaluation_target, test_data)
            result = Result(x_train, x_test, y_train, y_test, y_preds, model_def, fitted_model, data)
            results.append(result)

            # for reporter in reporters:
            #     reporter.update(result)
    return results


def build_and_package_model(model_def, data, data_description=None, evaluate=False,
                            reporters=None, prep_index=None, train_index=None):
    x_train, y_train, fitted_model = fit_model(model_def, data, prep_index, train_index)
    y_preds = predict(fitted_model, x_train)
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
