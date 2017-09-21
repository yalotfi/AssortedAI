import numpy as np

from assort.cost_functions import mse

# Implement Linear Regression trained with SGD
# h(x) = theta.T * X
# J(theta) = (1 / 2) * sum((h(theta) - y).^2)
# grad_J(theta) = sum()


def compute_cost(X, y, theta):
    # m = y.size
    X = np.c_[np.ones((X.shape[0], 1)), X]
    y_hat = np.dot(X, theta)
    cost = mse(y_hat, y)
    grad = mse(y_hat, y, derivative=True, X=X)
    return (cost, grad)


class LinearRegression(object):
    """OLS regression trained with Stochastic Gradient Descent

    Attributes:
        hyperparamters -- python dictionary
        X_train -- feature matrix of (m_examples, n_features)
        y_train -- label vector of (m_examples, 1)
        theta -- parameter vector of (n_features, 1)
        cost_cache -- python list of historical training error

    Methods:
        train_sgd -- perform stochastic gradient descent

    """

    def __init__(self, hyperparamters):
        self.alpha = hyperparamters["learning_rate"]
        self.epochs = hyperparamters["training_iters"]
        self.cost_cache = []
        self.trained_params = {}

    def hypothesis(self, X, theta):
        return np.dot(X, theta)

    def train_sgd(self, X_train, y_train):
        """Fit OLS Regression with Stochastic Gradient Descent

        Arguments:
            X_train -- training feature matrix with shape (m, n)
            y_train -- training label vector with shape (m, 1)

        Return:
            self
        """
        n = X_train.shape[1]
        theta = np.zeros((n + 1, 1))
        X = np.c_[np.ones((theta.shape)), X_train]
        for i in range(self.epochs):
            # 1) Make prediction
            y_hat = self.hypothesis(X, theta)
            # 2) Compute error of prediction and store it in cache
            cost = mse(y_hat, y_train)
            self.cost_cache.append(cost)
            # 3) Update parameters, theta, by learning rate and gradient
            grads = mse(y_hat, y_train, derivative=True, X=X)
            theta -= self.alpha * grads
        # Save optimized parameters
        self.trained_params = {"thetas": theta}
        return self
