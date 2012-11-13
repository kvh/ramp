from sklearn import hmm


class Probabilities(object):
    """ wraps a scikit learn estimator to return probabilities (if
    it supports it)
    """
    def __init__(self, estimator):
        self.estimator = estimator

    def __str__(self):
        return u"Probabilites for %s" % self.estimator

    def fit(self, x, y):
        self.estimator.fit(x,y)

    def predict(self, x):
        probs = self.estimator.predict_proba(x)
        return probs
