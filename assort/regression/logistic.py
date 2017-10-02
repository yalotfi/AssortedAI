import numpy as np

from assort.activations import sigmoid
from assort.cost_functions import cross_entropy


class LogisticRegression(object):
    """Logistic regression trained on batch Gradient Descent

    Attributes:
        hyperparams -- python dictionary defining model hyperparameters
                "training_iters": int, ideally multiple of 100
                "learning_rate": float32, scales parameter update rule
                "init_param_bound": float32, range in which to init weights
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
    def __init__(self, hyperparamters):
        self.hyperparamters = hyperparamters
        self.cost_cache = []
        self.trained_params = {}
        self.trained_grads = {}

    def _hypothesis(self, X, w, b):
        Z = np.dot(X, w) + b
        return sigmoid(Z)

    def gradient_descent(self, X_train, y_train, print_cost_freq=100):
        """Fit model to training data with stochastic gradient descent

        Arguments:
            X_train -- train set, feature matrix (m, n)
            y_train -- train set, label vector (m, 1)
            print_cost_freq -- how often to print cost

        Return:
            self
        """
        # Define helper variables: iters, learning rate, dim, params
        alpha = self.hyperparamters['learning_rate']
        epochs = self.hyperparamters['training_iters']
        m, n = X_train.shape[0], X_train.shape[1]
        w, b = self.init_params(n)

        # Training with gradient descent
        print("Training model...")
        for i in range(epochs):
            # Forward pass computes prediction and its loss
            y_hat =  self._hypothesis(X_train, w, b)
            cost = cross_entropy(y_hat, y_train)

            # Backward pass computes gradient of loss w respect to each param
            grads = cross_entropy(y_hat, y_train, derivative=True, X=X_train)

            # Assertions gradient and paramter dims
            assert(grads['dw'].shape == w.shape)
            assert(grads['db'].dtype == float)
            assert(cost.shape == ())

            # Update rule for tweaking parameters
            w = w - alpha * grads['dw']
            b = b - alpha * grads['db']

            # Record model error every 100 iterations
            if i % print_cost_freq == 0:
                self.cost_cache.append(cost)
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

    def init_params(self, n):
        bound = self.hyperparamters['init_param_bound']
        w = np.random.randn(n, 1) * bound
        b = np.zeros((1, 1))
        return w, b
