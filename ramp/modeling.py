from ramp.builders import build_featureset_safe, build_target_safe, apply_featureset_safe, apply_target_safe
from ramp.estimators.base import FittedEstimator
from ramp.result import Result
from ramp.store import Storable


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

    def __init__(self, fitted_model, data, data_description, evaluation_metrics=None, reports=None):
        super(PackagedModel, self).__init__()
        self.fitted_model = fitted_model
        self.data_rows = len(data)
        self.data_columns = len(data.columns)
        self.data_description = data_description
        self.evaluation_metrics = evaluation_metrics
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


def cross_validate(model_def, data, folds, evaluation_metrics=None, reporters=None, repeat=None):
    """
    """
    results = []
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
            x_test, y_true, y_preds = predict_model(model_def, data, fitted_model)
            result = Result(model_def, x_train, y_train, x_test, y_test, y_preds, fitted_model, evaluation_metrics)
            results.append(result)
            #TODO
            ### reporter/metrics work here
    return results, metrics, reports


def build_and_package_model(model_def, data, data_description, evaluation_metrics=None,
                            reporters=None, train_index=None, prep_index=None):
    x_train, y_train, fitted_model = model_fit(model_def, data, prep_index, train_index)

    # TODO
    eval_ = evaluation_metrics
    reports = []

    packaged_model = PackagedModel(fitted_model,
                                   data,
                                   data_description,
                                   evaluation_metrics,
                                   reports)
    return packaged_model