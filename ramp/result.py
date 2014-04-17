from ramp.store import Storable

class Result(Storable):
    def __init__(self, x_train, x_test, y_train, y_test, y_preds, model_def, fitted_model, original_data):
        """
        Class for storing the result of a single model fit.
        """
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_preds = y_preds
        self.model_def = model_def
        self.fitted_model = fitted_model
        self.original_data = original_data

