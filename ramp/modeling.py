from ramp.builders import build_featureset_safe, build_target_safe, apply_featureset_safe, apply_target_safe
from ramp.estimators.base import FittedEstimator


class FittedModel(object):

    def __init__(self,
                 model_def,
                 fitted_features,
                 fitted_target,
                 fitted_estimator):
        self.model_def = model_def
        self.fitted_features = fitted_features
        self.fitted_target = fitted_target
        self.fitted_estimator = fitted_estimator


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