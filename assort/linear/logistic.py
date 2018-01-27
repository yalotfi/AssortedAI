import numpy as np

from assort import _INITIALIZER_CONFIG
from assort.activations import sigmoid
from assort.activations import softmax
from assort.regularizers import l2_reg


class LinearClassifier(object):
    """Base class for Linear Classifiers"""

    def __init__(self, epochs=1, lr=0.05, lmda=0.01):
        super(LinearClassifier, self).__init__()
        # Take in model hyperparameters at object instantiation
        self._epochs = epochs
        self._alpha = lr
        self._lmda = lmda

        # Define hyperparameter dictionary as attribute
        self.hyperparameters = {
            "epochs": self._epochs,
            "learning_rate": self._alpha,
            "regularization_term": self._lmda
        }

        # Training attributes
        self.cost_cache = []
        self.trained_params = {}
        self.trained_grads = {}

    def _init_zeros(self, n, k):
        """
        Initialize model parameters with zero

        Arguments
        ---------
        n : int
            Define number of features, n
        k : int
            Define number of classes, k

        Returns
        -------
        ndarray
            Initialized weights, w, with shapes (n, k)
        float
            Initialized bias, b
        """
        w = np.zeros((n, k))
        b = 0.
        return w, b

    def _binary_xent(self, Y_hat, Y):
        """
        Binary Cross Entropy loss function

        Arguments
        ---------
        Y_hat : ndarray
            Probability of y given x
        Y : ndarray
            Actual value of y given x

        Returns
        -------
        float
            Cost (or loss) of predicted values
        """
        m = Y.shape[0]
        case_0 = Y * np.log(Y_hat)
        case_1 = (1 - Y) * np.log(1 - Y_hat)
        return -(1 / m) * np.sum(case_0 + case_1)

    def _categorical_xent(self, Y_hat, Y):
        """
        Categorical Cross Entropy loss function

        Arguments
        ---------
        Y_hat : ndarray
            Probability of y given x
        Y : ndarray
            Actual value of y given x

        Returns
        -------
        float
            Cost (or loss) of predicted values
        """
        m = Y.shape[0]
        logprobs = np.log(Y_hat)
        return -(1 / m) * np.sum(Y * logprobs)

    def _batch_gradient_descent(self, propagate, X, y):
        """
        Perform the batch gradient descent algorithm

        Arguments
        ---------
        propagate : callback
            Function that computes the cost and gradient of a single pass
        X : ndarray
            Training features
        y : ndarray
            Trainging labels
        """
        # Helpers: dimensionality and classes
        m, n = X.shape[0], X.shape[1]
        k = y.shape[1]

        # Initialize model parameters (weights and bias)
        w, b = self._init_zeros(n, k)

        cost_cache = []
        for i in range(self._epochs):
            # Perform single pass of forward and backward propagation
            cost, grads = propagate(X, y, w, b)

            # Store the cost for each iteration
            cost_cache.append(cost)
            if i % 100 == 0:
                print("Cost after iteration {}: {}".format(i, cost))

            # Update model parameters
            dw = grads["dw"]
            db = grads["db"]
            w = w - self._alpha * dw
            b = b - self._alpha * db

        # Store trained parameters and their gradients
        parameters = {"w": w, "b": b}
        gradients = {"dw": dw, "db": db}
        return parameters, gradients, cost_cache


class LogisticRegression(LinearClassifier):
    """
    Logistic regression trained on Stochastic Gradient Descent

    Arguments
    ---------
    epochs : int
        Number of full passes to make over dataset when training
    lr : float
        Learning rate which determines size of parameter update when training
    lmda : float
        Degree to which training cost should be regularized

    Attributes
    ----------
    hyperparameters : dictionary
        Stored hyperparameters for logging
    trained_params : dictionary
        Trained parameters
    trained_grads : dictionary
        Trained parameter gradients

    Methods
    -------
    fit
        Train model with batch gradient descent
    predict
        Make predictions for input data
    evaluate
        Compute the mean accuracy measure for the model
    """

    def __init__(self, epochs=1, lr=0.05, lmda=0.01):
        super().__init__(epochs, lr, lmda)

    def _hypothesis(self, X, w, b):
        Z = np.dot(X, w) + b
        return sigmoid(Z)

    def _propagate(self, X, Y, w, b):
        # Forward Pass
        A = self._hypothesis(X, w, b)
        cost = self._binary_xent(A, Y)

        # Backward Pass
        m = X.shape[0]
        dZ = A - Y
        dw = (1 / m) * np.dot(X.T, dZ)
        db = (1 / m) * np.sum(dZ)

        # Regularize the cost and gradient
        cost += l2_reg(w, self._lmda)
        dw += l2_reg(w, self._lmda, derivative=True)

        # Return the cost and gradients
        grads = {"dw": dw, "db": db}
        return cost, grads

    def fit(self, X, y):
        params, grads, cost_cache = self._batch_gradient_descent(
            self._propagate, X, y)

        # Store trained parameters, their gradients and cost history
        self.cost_cache = cost_cache
        self.trained_params = {"w": params["w"], "b": params["b"]}
        self.trained_grads = {"dw": grads["dw"], "db": grads["db"]}
        return self

    def predict(self, X, thresh=0.5):
        """
        Predict classes given input data

        This method uses the trained parameters learned during training. Use
        after fitting the model! A boolean array is return comparing the
        predicted probability to the given threshold.

        Arguments
        ---------
        X : ndarray
            Input data, X, to predict

        Returns
        -------
        bool ndarray
            Predicted classes for each input feature, X
        """
        # Make a prediction about each class
        w = self.trained_params["w"]
        b = self.trained_params["b"]
        y_pred = self._hypothesis(X, w, b) > thresh
        return y_pred.astype(int).reshape((y_pred.shape[0], 1))

    def evaluate(self, X_test, y_test):
        """
        Compute the mean accuracy of the trained classifier

        This method uses the trained parameters learned during training. Use
        after training! Furthermore, y_test should not be one-hot encoded to
        match the predictions which collapse to a vector as well.

        Arguments
        ---------
        X_test : ndarray
            Test features to evaluate of shape (m, n)
        y_test : ndarray
            Test labels to evaluate of shape (m, 1)

        Returns
        -------
        float
            Ratio of correct labels to incorrect labels
        """
        y_pred = self.predict(X_test)
        return np.mean(y_pred == y_test)


class SoftmaxRegression(LinearClassifier):
    """
    Softmax regression trained with Batch Gradient Descent

    Arguments
    ---------
    epochs : int
        Number of full passes to make over dataset when training
    lr : float
        Learning rate which determines size of parameter update when training
    lmda : float
        Degree to which training cost should be regularized

    Attributes
    ----------
    hyperparameters : dictionary
        Stored hyperparameters for logging
    trained_params : dictionary
        Trained parameters
    trained_grads : dictionary
        Trained parameter gradients

    Methods
    -------
    fit
        Train model with batch gradient descent
    predict
        Make predictions for input data
    evaluate
        Compute the mean accuracy measure for the model
    """

    def __init__(self, epochs=1, lr=0.05, lmda=0.01):
        super().__init__(epochs, lr, lmda)

    def _hypothesis(self, X, w, b):
        Z = np.dot(X, w) + b
        return softmax(Z, axis=1)

    def _propagate(self, X, Y, w, b):
        # Forward Pass
        A = self._hypothesis(X, w, b)
        cost = self._categorical_xent(A, Y)

        # Backward Pass
        dZ = A - Y
        dw = np.dot(X.T, dZ)
        db = np.sum(dZ)

        # Regularize the cost and gradient
        cost += l2_reg(w, self._lmda)
        dw += l2_reg(w, self._lmda, derivative=True)

        # Return the cost and gradients
        grads = {"dw": dw, "db": db}
        return cost, grads

    def fit(self, X, y):
        params, grads, cost_cache = self._batch_gradient_descent(
            self._propagate, X, y)

        # Store trained parameters, their gradients and cost history
        self.cost_cache = cost_cache
        self.trained_params = {"w": params["w"], "b": params["b"]}
        self.trained_grads = {"dw": grads["dw"], "db": grads["db"]}
        return self

    def predict(self, X):
        """
        Predict classes given input data

        This method uses the trained parameters learned during training. Use
        after training!

        Arguments
        ---------
        X : ndarray
            Input data, X, to predict

        Returns
        -------
        ndarray
            Predicted classes for each input feature, X
        """
        w = self.trained_params["w"]
        b = self.trained_params["b"]
        A = self._hypothesis(X, w, b)
        y_pred = np.argmax(A, axis=1)
        return y_pred.reshape((y_pred.shape[0], 1))

    def evaluate(self, X_test, y_test):
        """
        Compute the mean accuracy of the trained classifier

        This method uses the trained parameters learned during training. Use
        after training! Furthermore, y_test should not be one-hot encoded to
        match the predictions which collapse to a vector as well.

        Arguments
        ---------
        X_test : ndarray
            Test features to evaluate of shape (m, n)
        y_test : ndarray
            Test labels to evaluate of shape (m, 1)

        Returns
        -------
        float
            Ratio of correct labels to incorrect labels
        """
        y_pred = self.predict(X_test)
        return np.mean(y_pred == y_test)
