import numpy as np


class Svd_Regressor:

    w = 0
    b = 0

    def __init__(self):
        pass

    def __bias(self, x):
        return np.column_stack([np.ones(len(x)), x])

    def train(self, x, y):
        x = self.__bias(x)
        u, s, v = np.linalg.svd(x, full_matrices=False)
        w = v.T @ np.linalg.inv(np.diag(s)) @ u.T @ y
        self.w = w[1:]
        self.b = w[0]

    def predict(self, x):
        return (x @ self.w) + self.b
