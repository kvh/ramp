from sklearn import hmm


class Probabilities(object):
    """
    Wraps a scikit learn estimator to return probabilities (if
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

    def __str__(self):
        return u"Probabilites for %s" % self.estimator

    def fit(self, x, y):
        self.estimator.fit(x,y)

    def predict(self, x):
        probs = self.estimator.predict_proba(x)
        if self.binary:
            return [p[1] for p in probs]
        return probs


class BinaryProbabilities(Probabilities):
    def __init__(self, estimator):
        super(BinaryProbabilities, self).__init__(estimator, binary=True)
