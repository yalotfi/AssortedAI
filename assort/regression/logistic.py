import numpy as np

from assort import _INITIALIZER_CONFIG
from assort.activations import sigmoid
from assort.activations import softmax
from assort.regularizers import l2_reg
from assort.cost_functions import BinaryCrossEntropy


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
        case_0 = Y * np.log(Y_hat)
        case_1 = (1 - Y) * np.log(1 - Y_hat)
        return -np.sum(case_true + case_false)

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
        logprobs = np.log(Y_hat)
        return -np.sum(Y * logprobs)

    def _batch_gradient_descent(propagate, X, y):
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
        k = Y.shape[1]

        # Initialize model parameters (weights and bias)
        w, b = self._init_zeros(n, k)

        cost_cache = []
        for i in range(self._epochs):
            # Perform single pass of forward and backward propagation
            cost, grads = propagate(X, y, w, b)
            dw = grads["dw"]
            db = grads["db"]

            # Update model parameters
            w = w - self._alpha * dw
            b = b - self._alpha * db

            # Store the cost for each iteration
            cost_cache.append(cost)
            if i % 100 == 0:
                print("Cost after iteration {}: {}".format(i, cost))

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

    def _propagate(self, X, Y, w, b):
            # Forward Pass
            Z = np.dot(X, w) + b
            A = sigmoid(Z)
            cost = self._binary_xent(A, Y)

            # Backward Pass
            dZ = A - Y
            dw = (1 / m) * np.dot(X, dZ.T)
            db = (1 / m) * np.sum(dZ)

            # Regularize the cost and gradient
            cost += l2_reg(w, self._lmda)
            dw += l2_reg(w, self._lmda, derivative=True)

            # Return the cost and gradients
            grads = {"dw": dw, "db": db}
            return grads, cost

    def fit(X, y):
        params, grads, cost_cache = self._batch_gradient_descent(
            self._propagate, X, y)

        # Store trained parameters, their gradients and cost history
        self.cost_cache = cost_cache
        self.trained_params = {"w": w, "b": b}
        self.trained_grads = {"dw": dw, "db": db}
        return self

    # def predict(X):
    #     """
    #     Predict classes given input data
    #
    #     This method uses the trained parameters learned during training. Use
    #     after training!
    #
    #     Arguments
    #     ---------
    #     X : ndarray
    #         Input data, X, to predict
    #
    #     Returns
    #     -------
    #     ndarray
    #         Predicted classes for each input feature, X
    #     """
    #     w = self.trained_params["w"]
    #     b = self.trained_params["b"]
    #     Y_hat = self._hypothesis(X, w, b)
    #     Y_pred = np.argmax(Y_hat, axis=1)
    #     return Y_pred
    #
    # def evaluate(X_test, y_test):
    #     """
    #     Compute the mean accuracy of the trained classifier
    #
    #     This method uses the trained parameters learned during training. Use
    #     after training! Furthermore, y_test should not be one-hot encoded to
    #     match the predictions which collapse to a vector as well.
    #
    #     Arguments
    #     ---------
    #     X_test : ndarray
    #         Test features to evaluate of shape (m, n)
    #     y_test : ndarray
    #         Test labels to evaluate of shape (m, 1)
    #
    #     Returns
    #     -------
    #     float
    #         Ratio of correct labels to incorrect labels
    #     """
    #     w = self.trained_params["w"]
    #     b = self.trained_params["b"]
    #     y_hat = np.argmax(self._hypothesis(X, w, b), axis=1)
    #     test_acc = np.mean(np.abs(y_hat - y_test))


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

    def _hypothesis(X, w, b):
        Z = np.dot(X, w) + b
        return softmax(Z)

    def _propagate(self, X, Y, w, b):
        # Forward Pass
        Y_hat = self._hypothesis(X, w, b)
        cost = self._categorical_xent(Y_hat, Y)

        # Backward Pass
        dZ = A - Y
        dw = np.dot(X, dZ.T)
        db = np.sum(dZ)

        # Regularize the cost and gradient
        cost += l2_reg(w, self._lmda)
        dw += l2_reg(w, self._lmda, derivative=True)

        # Return the cost and gradients
        grads = {"dw": dw, "db": db}
        return grads, cost

    def fit(X, y):
        params, grads, cost_cache = self._batch_gradient_descent(
            self._propagate, X, y)

        # Store trained parameters, their gradients and cost history
        self.cost_cache = cost_cache
        self.trained_params = {"w": w, "b": b}
        self.trained_grads = {"dw": dw, "db": db}
        return self

    def predict(X):
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
        Y_hat = self._hypothesis(X, w, b)
        Y_pred = np.argmax(Y_hat, axis=1)
        return Y_pred

    def evaluate(X_test, y_test):
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
        w = self.trained_params["w"]
        b = self.trained_params["b"]
        y_hat = np.argmax(self._hypothesis(X, w, b), axis=1)
        test_acc = np.mean(np.abs(y_hat - y_test))


class LogisticRegression(object):
    """Logistic regression trained on Stochastic Gradient Descent

    Attributes:
        X_train -- train set, feature matrix (m, n)
        y_train -- train set, label vector (m, k)
        cost_cache -- python list storing historical training error
        trained_params -- dictionary storing optimized params
        trained_grads -- dictionary storing optimized gradients

    Methods:
        ### ~WIP~ ###
        gradient_descent -- Train model with gradient descent
        predict -- make binary predictions with trained model
        ### TO-DO ###
        accuracy -- provide accuracy metrics for the model
        plot_cost -- plot training error over number of training iterations
        #############
    """

    def __init__(self,
                 weight_initializer='random_normal',
                 bias_initializer='zeros'):
        # Set paramter initializers
        if weight_initializer in _INITIALIZER_CONFIG:
            self.weight_initializer = _INITIALIZER_CONFIG[weight_initializer]()
        if bias_initializer in _INITIALIZER_CONFIG:
            self.bias_initializer = _INITIALIZER_CONFIG[bias_initializer]()

        # Assigned when model is training
        self.cost_cache = []
        self.trained_params = {}
        self.trained_grads = {}

    def gradient_descent(self, X, y, alpha, epochs, print_cost_freq=10):
        """Fit model to training data with stochastic gradient descent

        Arguments:
            alpha -- Learning rate
            epochs -- Training iterations
            print_cost_freq -- how often to print cost

        Return:
            self
        """
        # Define helper variables: iters, learning rate, dim, params
        m, n = X.shape[0], X.shape[1]
        k_classes = y.shape[1]
        w = self.weight_initializer((n, 1))
        b = self.bias_initializer((1, k_classes))

        # Training with gradient descent
        print("Training model...")
        for i in range(epochs):
            # 1) Make prediction
            y_hat = sigmoid(np.dot(X, w) + b)
            # 2) Compute loss and gradients of parameters
            bce = BinaryCrossEntropy(y, y_hat, X)
            cost = bce.get_cost
            grads = bce.get_grads
            # 3) Update weights and bias
            w = w - alpha * grads['dw']
            b = b - alpha * grads['db']
            # 4) Save and print cost after every training iteration
            self.cost_cache.append(cost)
            if i % print_cost_freq == 0:
                print('Error after {} epochs: {}'.format(i, cost))

        # Store optimized parameters
        self.trained_params = {
            'w': w,
            'b': b
        }
        # Store optimized gradients
        self.trained_grads = {
            'dw': grads['dw'],
            'db': grads['db']
        }
        print("Model is trained, optimized results stored...\n")
        return self

    def predict(self, X_test, y_test, binary_threshold=0.5):
        """Predict binary labels on a given data set

        Arguments:
            X_test -- test set: feature matrix (n_features, m_test)
            y_test -- test set: label vector (1, m_test)
            binary_threshold -- probability threshold to pick binary class
        """
        # Define helper variables
        m = X_test.shape[1]  # training examples to sum over
        preds = np.zeros((1, m))  # init empty prediction array
        w = self.trained_params['w']  # pull trained params
        b = self.trained_params['b']  # pull trained gradients
        y_hat = sigmoid(np.dot(w.T, X_test) + b)  # make predictions

        # Take probability vector, y_hat: output predicted classes
        for i in range(y_hat.shape[0]):
            if y_hat[0, i] <= binary_threshold:
                preds[0, i] = 0
            elif y_hat[0, i] > binary_threshold:
                preds[0, i] = 1

        # Assess prediction
        score = self.accuracy(y_test, preds)

        return preds, score

    def accuracy(self, y_test, preds):
        """Assess accuracy of predictions from trained model"""
        pass

    @property
    def get_weights(self):
        return self.trained_params

    @property
    def get_gradients(self):
        return self.trained_grads


class SoftmaxRegression(LogisticRegression):
    """docstring for SoftmaxRegression."""
    def __init__(self,
                 hyperparameters,
                 weight_initializer='random_normal',
                 bias_initializer='zeros'):
        super().__init__(hyperparameters, weight_initializer, bias_initializer)

    def print_model(self, X_train, y_train):
        alpha = self.hyperparameters['learning_rate']
        epochs = self.hyperparameters['training_iters']
        m, n = X_train.shape[0], X_train.shape[1]
        k_classes = y_train.shape[1]
        w = self.weight_initializer((n, 1))
        b = self.bias_initializer((k_classes, 1))
        print("\nPrinting Model Configuration:")
        print("Learning Rate: ", alpha)
        print("Training Iterations: ", epochs)
        print("Training Examples: ", m)
        print("Features: ", n)
        print("k-Classes: ", k_classes)
        print("Weight Dims: ", w.shape)
        print("Bias Dims: ", b.shape)
