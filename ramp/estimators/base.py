import types

__all__ = ['Wrapper', 'Estimator', 'FittedEstimator', 'Probabilities', 'BinaryProbabilities']


class Wrapper(object):
    def __init__(self,obj):
        self._obj = obj
    
    def __getattr__(self, attr):
        
        if hasattr(self._obj, attr):
            attr_value = getattr(self._obj,attr)
            
            if isinstance(attr_value,types.MethodType):
                def callable(*args, **kwargs):
                    return attr_value(*args, **kwargs)
                return callable
            else:
                return attr_value
            
        else:
            raise AttributeError


class Estimator(Wrapper):
    def __init__(self, estimator):
        self.estimator = estimator
        super(Estimator, self).__init__(estimator)

    def fit(self, x, y):
        return self.estimator.fit(x.values, y.values)

    def predict(self, x):
        return self.estimator.predict(x.values)


class FittedEstimator(Wrapper):
    def __init__(self, fitted_estimator, x_train, y_train):
        # compute metadata
        self.fitted_estimator = fitted_estimator
        super(FittedEstimator, self).__init__(fitted_estimator)


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
        if self.binary:
            return [p[1] for p in probs]
        return probs


class BinaryProbabilities(Probabilities):
    def __init__(self, estimator):
        super(BinaryProbabilities, self).__init__(estimator, binary=True)
