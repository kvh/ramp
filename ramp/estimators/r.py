import numpy as np
from rpy2.robjects import FloatVector
from rpy2.robjects.packages import importr
from rpy2 import robjects
stats = importr('stats')
base = importr('base')


def matrix_to_r_dataframe(x):
        rx = FloatVector(np.ravel(x))
        rx = robjects.r['matrix'](rx, nrow = len(x), byrow=True)
        return robjects.r["data.frame"](rx)


class REstimator(object):
    def __init__(self, r_estimator, **kwargs):
        self.estimator = r_estimator
        self.kwargs = kwargs

    def fit(self, x, y):
        rx = matrix_to_r_dataframe(x)
        ry = FloatVector(y)
        robjects.globalenv["y"] = ry
        self.estimator_fit = self.estimator("y ~ .",  data=rx,
                **self.kwargs)

    def predict(self, x):
        rx = matrix_to_r_dataframe(x)
        return np.array(stats.predict(self.estimator_fit, rx)[0])


class OrderedLogit(object):
    def fit(self, x, y):
        ordinal = importr('ordinal')
        rx = matrix_to_r_dataframe(x)
        self.levels = range(int(round(min(y))), int(round(max(y)))+1)
        ry = base.factor(FloatVector(y), levels=self.levels, ordered=True)
        robjects.globalenv["y"] = ry
        self.clmfit = ordinal.clm("y ~ .",  data=rx)
        #print base.summary(self.clmfit)

    def predict(self, x):
        rx = matrix_to_r_dataframe(x)
        rfac = stats.predict(self.clmfit, rx, type="class")[0]
        rvec = [self.levels[v - 1] for v in rfac]
        return rvec


class WeightedLM(object):
    def fit(self, x, y, weights):
        rx = matrix_to_r_dataframe(x)
        ry = FloatVector(y)
        rw = FloatVector(weights)
        robjects.globalenv["score"] = ry
        self.lmfit = stats.lm("score ~ .",  data=rx, weights=rw)
        #print base.summary(self.clmfit)

    def predict(self, x):
        rx = matrix_to_r_dataframe(x)
        rvec = stats.predict(self.lmfit, rx)[0]
        return np.array(rvec)


class GBM(REstimator):
    def __init__(self, **kwargs):
        gbm = importr('gbm')
        super(GBM, self).__init__(gbm.gbm, **kwargs)

