import numpy as np

from ..activations import softmax
from ..cost_functions import cross_entropy


class SoftmaxRegression(object):
    """Generalized logistic regression for K classes
    """
    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters
        self.cost_cache = []
        self.trained_params = {}
        self.trained_grads = {}

    def _init_params_zeros(self, dims):
        w = np.zeros((dims))
        b = 0
        return {"w": w, "b": b}

    def _hypothesis(self, X, parameters):
        """Hypothesis function for softmax regression
        X: (m, n)
        w: (n, k)
        b: (k, 1)
        """
        Z = np.dot(X.T, parameters["w"]) + parameters["b"]
        return softmax(Z)

    def _compute_cost(self, y_hat, y):
        """Compute the loss between the predicted labels and the actual labels
        y_hat: (m, k)
        y: (m, k)
        """
        m = y_hat.shape[0]
        return -(1 / m) * np.sum(y.T * np.log(y_hat))

    def _compute_grad(self, X, y, y_hat):
        """Compute the gradient of the loss with respect to each parameter
        """
        m = y_hat.shape[0]
        dw = (1 / m) * np.dot(X, (y - y_hat).T)
        db = (1 / m) * np.sum(y - y_hat)
        return {"dw": dw, "db": db}

    def _propagate(self, X, y, params):
        y_hat = self._hypothesis(X, params)
        cost = self._compute_cost(y_hat, y)
        grads = self._compute_grad(X, y, y_hat.T)
        return cost, grads

    def gradient_descent(self, X, y, alpha=0.01):
        n, m = X.shape[0], X.shape[1]
        init_params = self._init_params_zeros((n, 1))
        for i in range(0, 1000):
            cost, grads = self._propagate(X, y, init_params)

            # Update params
            init_params["w"] = init_params["w"] - alpha * grads["dw"]
            init_params["b"] = init_params["b"] - alpha * grads["db"]

            # Store cost
            if i % 100 == 0:
                print(cost)
                self.cost_cache.append(cost)
