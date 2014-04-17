from ramp.store import Storable

class Result(Storable):
    def __init__(self, original_data, model_def, y_test, y_preds, x_test, x_train, y_train, prep_data, fitted_model, evals):
        """
        Class for storing the result of a single model fit.
        """
        self.original_data = original_data
        self.model_def = model_def
        self.y_test = y_test
        self.y_preds = y_preds
        self.x_test = x_test
        self.x_train = x_train
        self.y_train = y_train
        self.prep_data = prep_data  # TODO touch base with Ken about this, which is used to get colnames
        self.fitted_model = fitted_model
        self.evals = evals

