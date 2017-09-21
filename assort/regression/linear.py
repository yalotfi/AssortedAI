import numpy as np

from assort.cost_functions import mse

# Implement Linear Regression trained with SGD
# h(x) = theta.T * X
# J(theta) = (1 / 2) * sum((h(theta) - y).^2)
# grad_J(theta) = sum()


def compute_cost(X, y, theta):
    # m = y.size
    y_hat = hypothesis(X, theta)
    cost = mse(y_hat, y)
    grad = mse(y_hat, y, derivative=True, X_train=X)
    return (cost, grad)


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
        train -- takes training examples and labels

    """

    def __init__(self, hyperparamters):
        self.alpha = hyperparamters["learning_rate"]
        self.epochs = hyperparamters["training_iters"]
        self.cost_cache = np.zeros(self.epochs)
        self.trained_params = {}
        self.trained_grads = {}

    def hypothesis(self, X, theta):
        return np.dot(X, theta)

    def train(self, X_train, y_train):
        # m = y_train.size
        theta = np.zeros(())
        for i in range(self.epochs):
            y_hat = self.hypothesis(X_train, theta)
            cost = mse(y_hat, y_train)
            grad = mse(y_hat, y_train, derivative=True, X=X_train)
            # dw = (1 / m) * np.dot(X_train.T, (y_hat - y_train))
            # db = (1 / m) * (y_hat - y_train)
            # assert(dw.shape == self.w.shape)
            # assert(db.dtype == float)
            assert(cost.shape == ())
            theta -= self.alpha * grad
        # self.trained_grads = {
        #     "dw": dw,
        #     "db": db
        # }
        # self.trained_params = {
        #     "w":

        #     "b"
        # }
