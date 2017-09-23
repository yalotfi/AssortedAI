import numpy as np
import matplotlib.pyplot as plt

from assort.cost_functions import mse


class LinearRegression(object):
    """OLS regression trained with Stochastic Gradient Descent

    Attributes:
        hyperparamters -- python dictionary storing:
            "learning_rate": float32, alpha that scaled parameter update
            "epochs": -- int, number of passes over training set
        cost_cache -- numpy array of historical training error
        trained_params -- python dictionary storing:
            "theta" -- ndarray - optimized model parameters

    Methods:
        hypothesis -- h(theta) function
        train_sgd -- perform stochastic gradient descent
        predict -- make prediction after fitting linear regresion
        plot_error -- plot cost after each training iteration
    """

    def __init__(self, hyperparamters):
        self.hyperparamters = hyperparamters
        self.cost_cache = np.zeros(self.hyperparamters["epochs"])
        self.trained_params = {}

    def hypothesis(self, X, theta):
        return np.dot(X, theta)

    def train_sgd(self, X_train, y_train, print_cost_freq=100):
        """Fit OLS Regression with Stochastic Gradient Descent

        Arguments:
            X_train -- training feature matrix with shape (m, n)
            y_train -- training label vector with shape (m, 1)
            print_cost_freq -- print cost when train iter mod freq = 0

        Return:
            self
        """
        # Initial housekeeping before running SGD
        alpha = self.hyperparamters["learning_rate"]
        epochs = self.hyperparamters["epochs"]
        m, n = X_train.shape[0], X_train.shape[1]
        theta = np.zeros((n + 1, 1))
        X = np.c_[np.ones((m, 1)), X_train]
        print("Initialized params to zero...\n")

        # Start running SGD
        print("Training model...")
        for i in range(epochs):
            # 1) Make prediction
            y_hat = self.hypothesis(X, theta)

            # 2) Compute error of prediction and store it in cache
            cost = mse(y_hat, y_train)
            self.cost_cache[i] = cost
            if i % print_cost_freq == 0:
                print("Error at iteration {}: {}".format(i, cost))

            # 3) Update parameters, theta, by learning rate and gradient
            grads = mse(y_hat, y_train, derivative=True, X=X)
            theta = theta - alpha * grads

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
        print("Plotting model error...\n")
        plt.plot(self.cost_cache)
        plt.ylabel('Training Cost')
        plt.xlabel('Training Iteration')
        plt.show()
