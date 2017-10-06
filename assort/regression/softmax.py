import numpy as np
import matplotlib.pyplot as plt

from assort.activations import softmax


class SoftmaxRegression(object):
    """Generalized logistic regression for K classes
    """
    def __init__(self):
        # self.hyperparameters = hyperparameters
        self.cost_cache = []
        self.trained_params = {}
        self.trained_grads = {}

    def _init_params_zeros(self, n, k):
        w = np.random.randn(n, k)
        b = np.zeros((k, 1))
        return {"w": w, "b": b}

    def _hypothesis(self, X, w, b):
        """Hypothesis function for softmax regression
        X: (n, m)  # (features, examples)
        w: (n, k)  # (features, classes)
        b: (k, 1)  # (classes, 1)
        """
        Z = np.dot(w.T, X) + b
        A = softmax(Z)
        return A

    def _compute_cost(self, A, Y):
        """Compute the loss between the predicted labels and the actual labels
        A: (k, m)  # "Activated" neuron, softmax-transform of linear function
        Y: (k, m)  # Actual labels that we compare probability distribution to
        """
        m = A.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A))
        return cost

    def _compute_grad(self, X, Y, A):
        """Compute the gradient of the loss with respect to each parameter
        """
        m = A.shape[0]
        dZ = Y - A
        dw = -(1 / m) * np.dot(X, dZ.T)
        db = -(1 / m) * np.sum(dZ)
        grads = {
            "dw": dw,
            "db": db
        }
        return grads

    def _propagate(self, X, Y, w, b):
        A = self._hypothesis(X, w, b)
        cost = self._compute_cost(A, Y)
        grads = self._compute_grad(X, Y, A)
        return cost, grads

    def gradient_descent(self, X, Y, alpha=0.00001):
        n, m = X.shape[0], X.shape[1]
        k = Y.shape[0]
        init_params = self._init_params_zeros(n, k)
        w, b = init_params["w"], init_params["b"]
        for i in range(0, 2000):
            # Forward and backward passes
            cost, grads = self._propagate(X, Y, w, b)
            dw, db = grads["dw"], grads["db"]
            # Update params
            w = w - alpha * dw
            b = b - alpha * db
            # Store cost
            if i % 100 == 0:
                print(cost)
                self.cost_cache.append(cost)
        self.trained_params = {"w": w, "b": b}
        self.trained_grads = {"dw": dw, "db": db}
        return self

    def show_error(self):
        plt.plot(self.cost_cache)
        plt.ylabel('Training Cost')
        plt.xlabel('Training Iteration')
        plt.show()
