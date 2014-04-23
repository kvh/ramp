import logging

from ramp.builders import build_featureset_safe, build_target_safe, apply_featureset_safe, apply_target_safe
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


def fit_model(model_def, data, prep_index=None, train_index=None):
    # create training set
    x_train, fitted_features = build_featureset_safe(model_def.features, data, prep_index, train_index)
    y_train, fitted_target = build_target_safe(model_def.target, data, prep_index, train_index)

    # fit estimator
    model_def.estimator.fit(x_train, y_train)

    # unnecesary?
    fitted_estimator = FittedEstimator(model_def.estimator, x_train, y_train)

    fitted_model = FittedModel(model_def, fitted_features, fitted_target, fitted_estimator)
    return x_train, y_train, fitted_model


def predict_model(model_def, predict_data, fitted_model, compute_actuals=True):
    # create test set and predict
    x_test = apply_featureset_safe(model_def.features, predict_data, fitted_model.fitted_features)
    if compute_actuals:
        y_test = apply_target_safe(model_def.target, predict_data, fitted_model.fitted_target)
    else:
        y_test = None
    y_preds = fitted_model.fitted_estimator.predict(x_test)
    return x_test, y_test, y_preds


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
                raise ValueError("Fold is not of right dimension (%d)"%len(fold))
            x_train, y_train, fitted_model = fit_model(model_def, data, prep_index, train_index)
            x_test, y_test, y_preds = predict_model(model_def, data, fitted_model)
            result = Result(x_train, x_test, y_train, y_test, y_preds, model_def, fitted_model, data)
            results.append(result)
            logging.debug(result.__dict__)
            
            for reporter in reporters:
                reporter.update(result)
    return results, reporters


def build_and_package_model(model_def, data, data_description, evaluate=False,
                            reporters=None, prep_index=None, train_index=None):
    x_train, y_train, fitted_model = fit_model(model_def, data, prep_index, train_index)
    result = None
    if evaluate:
        # only evaluate on train (this seems reasonable)
        y_preds = fitted_model.fitted_estimator.predict(x_train)
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
