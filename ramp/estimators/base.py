import types

__all__ = ['Wrapper', 'Estimator', 'FittedEstimator',
           'Probabilities', 'BinaryProbabilities', 'wrap_sklearn_like_estimator']


class Wrapper(object):
    def __init__(self, obj):
        self._obj = obj

    def __getattr__(self, attr):

        if hasattr(self._obj, attr):
            attr_value = getattr(self._obj,attr)

            if isinstance(attr_value, types.MethodType):
                def callable(*args, **kwargs):
                    return attr_value(*args, **kwargs)
                return callable
            else:
                return attr_value

        else:
            raise AttributeError

    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)


class Estimator(Wrapper):
    def __init__(self, estimator):
        self.estimator = estimator
        super(Estimator, self).__init__(estimator)

    def __repr__(self):
        return repr(self.estimator)

    def fit(self, x, y, **kwargs):
        return self.estimator.fit(x.values, y.values, **kwargs)

    def predict_maxprob(self, x, **kwargs):
        """
        Most likely value. Generally equivalent to predict.
        """
        return self.estimator.predict(x.values, **kwargs)

    def predict(self, x, **kwargs):
        """
        Model output. Not always the same as scikit_learn predict. E.g., in the
        case of logistic regression, returns the probability of each outome.
        """
        return self.estimator.predict(x.values, **kwargs)


class Probabilities(Estimator):
    """
    Wraps a scikit-learn-like estimator to return probabilities (if
    it supports it)
    """
    def __init__(self, estimator, binary=False):
        """
        binary: If True, predict returns only the probability
            for the positive class. If False, returns probabilities for
            all classes.
        """
        self.estimator = estimator
        self.binary = binary
        super(Probabilities, self).__init__(estimator)

    def __str__(self):
        return u"Probabilites for %s" % self.estimator

    def predict(self, x):
        probs = self.estimator.predict_proba(x)
        if probs.shape[1] == 2 or self.binary:
            return probs[:,1]
        return probs


class BinaryProbabilities(Probabilities):
    def __init__(self, estimator):
        super(BinaryProbabilities, self).__init__(estimator, binary=True)


class FittedEstimator(Wrapper):
    def __init__(self, fitted_estimator, x_train, y_train):
        # compute metadata
        self.fitted_estimator = fitted_estimator
        super(FittedEstimator, self).__init__(fitted_estimator)


def wrap_sklearn_like_estimator(estimator):
    if isinstance(estimator, Estimator):
        return estimator
    elif estimator is None:
        return None
    elif not (hasattr(estimator, "fit") and (hasattr(estimator, "predict")
                                          or hasattr(estimator, "predict_proba"))):
        raise ValueError, "Invalid estimator: %s" % estimator
    elif hasattr(estimator, "predict_proba"):
        return Probabilities(estimator)
    else:
        return Estimator(estimator)

