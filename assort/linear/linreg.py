import numpy as np
import matplotlib.pyplot as plt

from assort import _INITIALIZER_CONFIG
from assort import _COST_FUNC_CONFIG
from assort.cost_functions import MeanSquaredError


class RegressionModel(object):
    """LinearModel: Base class for all linear models.
    """

    def __init__(self, X_train, y_train):
        super(RegressionModel, self).__init__()
        # Get training features and labels
        self.X_train = X_train
        self.y_train = y_train
        self.m, self.n = self.X_train.shape[0], self.X_train.shape[1]

        # Assigned when model is training
        self.cost_cache = []
        self.trained_params = {}
        self.trained_grads = {}

    def _init_weights(self, weight_initializer, shape):
        try:
            weight_initializer = _INITIALIZER_CONFIG[weight_initializer]()
            return weight_initializer(shape)
        except ValueError:
            print("Initializer not supported...")

    def _init_bias(self, bias_initializer, shape):
        try:
            bias_initializer = _INITIALIZER_CONFIG[bias_initializer]()
            return bias_initializer(shape)
        except ValueError:
            print("Initializer not supported...")

    def _init_cost_func(self, objective, y_train, y_hat, X_train):
        try:
            return _COST_FUNC_CONFIG[objective](y_train, y_hat, X_train)
        except ValueError:
            print("Objective not supported...")

class LinearRegression(RegressionModel):
    """OLS regression trained with Stochastic Gradient Descent

    Attributes:
        X_train -- training feature matrix with shape (m, n)
        y_train -- training label vector with shape (m, 1)
        weight_initializer -- how to initialize model parameters
        cost_cache -- numpy array of historical training error
        trained_params -- python dictionary storing:
            "theta" -- ndarray - optimized model parameters

    Methods:
        fit -- perform gradient descent
        predict -- make prediction after fitting linear regresion
        plot_error -- plot cost after each training iteration
    """

    def __init__(self,
                 X_train,
                 y_train,
                 objective='mean_squared_error',
                 weight_initializer='zeros'):
        super().__init__(X_train, y_train)
        intercept = np.ones((self.m, 1))
        self.X_ = np.c_[intercept, self.X_train]
        self.theta_start = self._init_theta(weight_initializer)
        self.objective = objective

    def _init_theta(self, weight_initializer):
        shape = (self.n + 1, 1)
        return self._init_weights(weight_initializer, shape)

    def _propagate(self, theta):
        y_hat = np.dot(self.X_, theta)
        mse = self._init_cost_func(
            self.objective, self.y_train, y_hat, self.X_)
        return (mse.get_cost, mse.get_grads)

    def fit(self, optimizer, print_cost_freq=100):
        """Fit OLS Regression with Stochastic Gradient Descent

        Arguments:
            optimizer -- optimizer class
            print_cost_freq -- print cost when train iter mod freq = 0

        Return:
            self
        """
        print("Training model...")
        w = self.theta_start
        for i in range(optimizer.epochs):
            cost, grad = self._propagate(w)
            # 3) Update parameters, theta, by learning rate and gradient
            w = optimizer._update_params(w, grad)
            # 4) Save and print cost after every training iteration
            self.cost_cache.append(cost)
            if i % print_cost_freq == 0:
                print("Error at iteration {}: {}".format(i, cost))

        # Save optimized parameters
        self.trained_params = {"theta": w}
        self.trained_grads = {"grads": grad}
        print("Model is trained, optimized results stored...\n")
        return self

    def predict(self, X):
        """Make a prediction with the trained model"""
        X_ = np.c_[np.ones((X.shape[0], 1)), X]
        theta = self.trained_params["theta"]
        return np.dot(X_, theta)

    def plot_error(self):
        """Simply plot the model error over training"""
        print("Plotting model error...\n")
        plt.plot(self.cost_cache)
        plt.ylabel('Training Cost')
        plt.xlabel('Training Iteration')
        plt.show()
