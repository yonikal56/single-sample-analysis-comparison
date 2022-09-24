import numpy as np
import math


class DOC:
    epsilon = math.pow(10, -4)

    def dkl(self, x, y):
        return sum([(x[k] * math.log(x[k] / y[k])) for k in range(len(x))])

    def get_shared(self, xi, xj):
        return [key for key in range(len(xi)) if xi[key] > self.epsilon and xj[key] > self.epsilon]

    def _get_dissimilarity(self, xi, xj, shared_keys=None):
        shared_keys = self.get_shared(xi, xj) if shared_keys is None else shared_keys
        shared_xi_sum = sum([xi[key] for key in shared_keys])
        shared_xj_sum = sum([xj[key] for key in shared_keys])
        xihat = np.array([xi[key] / shared_xi_sum for key in shared_keys])
        xjhat = np.array([xj[key] / shared_xj_sum for key in shared_keys])
        m = (xihat + xjhat) / 2
        return math.sqrt((self.dkl(xihat, m) + self.dkl(xjhat, m)) / 2)

    def _get_overlap(self, xi, xj, shared_keys=None):
        shared_keys = self.get_shared(xi, xj) if shared_keys is None else shared_keys
        return sum([(xi[key] + xj[key]) / 2 for key in shared_keys])

    def get_dissimilarity_overlap_point(self, xi, xj):
        xi = np.array(xi) / sum(xi)
        xj = np.array(xj) / sum(xj)
        shared_keys = self.get_shared(xi, xj)
        return [self._get_dissimilarity(xi, xj, shared_keys), self._get_overlap(xi, xj, shared_keys)]
