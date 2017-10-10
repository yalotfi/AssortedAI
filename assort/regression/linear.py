import numpy as np
import matplotlib.pyplot as plt

from assort import _INITIALIZER_CONFIG
from assort.cost_functions import MeanSquaredError


class LinearRegression(object):
    """OLS regression trained with Stochastic Gradient Descent

    Attributes:
        X_train -- training feature matrix with shape (m, n)
        y_train -- training label vector with shape (m, 1)
        weight_initializer -- how to initialize model parameters
        cost_cache -- numpy array of historical training error
        trained_params -- python dictionary storing:
            "theta" -- ndarray - optimized model parameters

    Methods:
        hypothesis -- h(theta) function
        gradient_descent -- perform gradient descent
        predict -- make prediction after fitting linear regresion
        plot_error -- plot cost after each training iteration
    """

    def __init__(self, X_train, y_train, weight_initializer='zeros'):
        # Get training features and labels
        self.X_train = X_train
        self.y_train = y_train

        # Set parameter initializer
        if weight_initializer in _INITIALIZER_CONFIG:
            self.weight_initializer = _INITIALIZER_CONFIG[weight_initializer]()

        # Assigned when model is training
        self.cost_cache = []
        self.trained_params = {}

    def gradient_descent(self, alpha, epochs, print_cost_freq=100):
        """Fit OLS Regression with Stochastic Gradient Descent

        Arguments:
            alpha -- Learning rate
            epochs -- Training iterations
            print_cost_freq -- print cost when train iter mod freq = 0

        Return:
            self
        """
        # Initial housekeeping before running SGD
        m, n = self.X_train.shape[0], self.X_train.shape[1]
        theta = self.weight_initializer((n + 1, 1))
        X_ = np.c_[np.ones((m, 1)), self.X_train]
        Y = self.y_train

        # Start running SGD
        print("Training model...")
        for i in range(epochs):
            # 1) Make prediction
            y_hat = np.dot(X_, theta)

            # 2) Compute error of prediction and store it in cache
            mse = MeanSquaredError(Y, y_hat, X_)
            cost = mse.get_cost
            self.cost_cache.append(cost)
            if i % print_cost_freq == 0:
                print("Error at iteration {}: {}".format(i, cost))

            # 3) Update parameters, theta, by learning rate and gradient
            theta = theta - alpha * mse.get_grad

        # Save optimized parameters
        self.trained_params = {"theta": theta}

        print("Model is trained...\n")
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
