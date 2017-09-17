import numpy as np

from assort.utils.cost_functions import mse

# Implement Linear Regression trained with SGD
# h(x) = theta.T * X
# J(theta) = (1 / 2) * sum((h(theta) - y).^2)
# grad_J(theta) = sum()


def hypothesis(X, theta):
    return np.dot(X, theta)


def compute_cost(X, y, theta):
    m = y.size
    y_hat = hypothesis(X, theta)
    cost1 = (1 / (2 * m)) * np.sum(np.square(y_hat - y))
    cost2 = mse(y_hat, y)
    return (cost1, cost2)


def compute_grads(theta, X, y):
    m = y.size
    y_hat = hypothesis(theta, X)
    return (1 / m) * np.sum(np.multiply(X, (y_hat - y)))


def sgd(theta, alpha):
    grad = compute_grads()
    return theta - alpha * grad


class LinearRegression(object):
    """OLS regression trained with Stochastic Gradient Descent

    Attributes:
        hyperparamters -- python dictionary
        X_train -- feature matrix of (m_examples, n_features)
        y_train -- label vector of (m_examples, 1)
        theta -- parameter vector of (n_features, 1)
        cost_cache -- python list of historical training error

    Methods:

    """

    def __init__(self, X_trian, y_train, hyperparamters):
        self.X = X_train
        self.y = y_train
        self.theta = self.init_params()
        self.cost_cache = []
        self.trained_params = {}
        self.trained_grads = {}

    @property
    def init_params(self):
        return np.zeros((self.X.shape[0], 1))
