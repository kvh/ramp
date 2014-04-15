
class Feature(object):
    def __init__(self, feature):
        self.feature = feature
        # Normalize(FillMissing('x1', 0))

    def __call__(self, data, prepped_feature=None, trained_feature=None):
        return self.apply(data, prepped_feature, trained_feature)

    def build(self, data, prep_data, train_data):
        data, _pf, _tf = self.feature.build(data, prep_data, train_data)
        ff = FittedFeature()
        ff.prepped_data = self.prepare(prep_data)
        ff.trained_data = self.train(train_data)
        feature_data = self.apply(data, pf, tf)
        return feature_data, ff

    def apply(self, data_ctx, fitted_feature=None):
        raise NotImplementedError

    def prepare(self, prep_data):
        pass

    def train(self, train_data):
        pass


class FittedFeature(Storable):
    def __init__(self, feature, fitted_features, train_index, prep_index, prepped_data=None, trained_data=None):
        # compute metadata
        self.train_n = len(train_data)
        self.prep_n = len(prep_data)
        self.train_data_key = key_from_index(train_data.index)
        self.prep_data_key = key_from_index(prep_data.index)
        ...
        self.* = *


class Estimator(object):
    def fit(self, x, y):
        pass

    def predict(self, x):
        pass


class FittedEstimator(Storable):
    def __init__ ():
        # compute metadata

    def predict
    def predict_proba


class FittedModel(Storable):

    def __init__(self,
                 model_def,
                 fitted_features,
                 fitted_target,
                 fitted_estimator):
        self.* = *


def build_featureset_safe(features, data, prep_data, train_data):
    pass

def evaluate(model_def, ):
    pass


class Result(object):
    def __init__(self, model_def, y_test, y_preds, x_test, x_train, y_train, fitted_model, evals):
        pass


def fit_model(model_def, train_data, prep_data):
    # create training set
    x_train, fitted_features = build_featureset_safe(model_def.features, train_data, prep_data, train_data)
    y_train, fitted_target = build_target_safe(model_def.target, train_data, prep_data, train_data)

    # fit estimator
    model_def.estimator.fit(x_train, y_train)

    # unnecesary?
    fitted_estimator = FittedEstimator(model_def.estimator)

    fitted_model = FittedModel(model_def, fitted_features, fitted_target, fitted_estimator)
    return x_train, y_train, fitted_model

def predict_model(model_def, fitted_model, predict_data, compute_actuals=True):
    # create test set and predict
    x_test = apply_featureset_safe(model_def.features, fitted_model.fitted_features, predict_data)
    if compute_actuals:
        y_test = apply_target_safe(model_def.target, test_data)
    else:
        y_test = None
    y_preds = fitted_model.fitted_estimator.predict(x_test)
    return x_test, y_preds, y_preds


def cv(df, model_def, folds, evaluation_metrics, reporters):

    for train_index, test_index, prep_index in folds:
        prep_data = align_index_safe(df, prep_index)
        train_data = align_index_safe(df, train_index)
        test_data = align_index_safe(df, test_index)
        
        x_train, y_train, fitted_model = fit_model(model_def, train_data, prep_data)

        x_test, y_test, y_preds = predict_model(model_def, fitted_model, test_data)

        # evaluate
        for metric in evaluation_metrics:
            eval = metric.evaluate(model_def, y_test, y_preds, x_test, fitted_model)

        result = Result(model_def, y_test, y_preds, x_test, fitted_models, evals)

        # report
        for r in reporters:
            r.update(result)

        results.append(result)
    return results, reporters



def align_index_safe(df, idx):
    # duplicate labels don't make sense in this context
    assert len(idx) == len(idx.unique()), "index contains %d duplicates" % len(idx) - len(idx.unique())
    df_idx = df.loc[idx]
    # ditto
    assert len(df_idx) == len(idx)
    return df_idx


"""
/model
    /features
        /feature_123Asf9A90
            prep.pkl
            train.pkl
        /feature_DFd8723DFd
            prep.pkl
            train.pkl
        ...
    /estimator_aslkfj893df
        train.pkl
    /target_2342dfas83D
        prep.pkl
        train.pkl
"""


def build_and_package_model(model_def, data, data_description, evaluation_metrics=None, reporters=None, train_index=None, prep_index=None):
    if train_index is not None:
        train_data = align_index_safe(data, train_index)
    else:
        train_data = data
    if prep_index is not None:
        prep_data = align_index_safe(data, prep_index)
    else:
        prep_data = data

    x_train, fitted_features = build_featureset_safe(model_def.features, data, prep_data, train_data)
    y_train, fitted_target = build_target_safe(model_def.target, data, prep_data, train_data)

    model_def.estimator.fit(x_train, y_train)
    fitted_estimator = model_def.estimator

    # unnecesary?
    fitted_estimator = FittedEstimator(model_def.estimator)
    fitted_model = FittedModel(model_def, fitted_features, fitted_target, fitted_estimator)

    eval_ = evaluation_metrics
    reporters = reports ....

    return fitted_model, reporters, eval_





