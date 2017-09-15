import numpy as np

from utils.activations import sigmoid
from utils.cost_functions import cross_entropy


def hypothesis(w, b, X):
    """Make prediction

    Args:
        w -- weight vector with shape (1, n)
        b -- bias scalar with shape (1, 1)
        X -- feature matrix with shape (

    Returns:
        y_hat -- linear predictions passed through sigmoid
    """
    return sigmoid(np.dot(w.T, X) + b)


def compute_grads(y_hat, y, X):
    """Compute gradient of the cost function

    Args:
        y_hat -- vector of predicted labels with shape (n, 1)
        y -- vector of actual labels with shape (n, 1)
        X -- feature matrix with shape (n_features, m_examples)

    Returns:
        grads -- python dictionary of weight and bias gradients
    """
    m = X.shape[1]
    dZ = y_hat - y
    dw = (1 / m) * np.dot(X, dZ.T)
    db = (1 / m) * np.sum(dZ)
    grads = {
        'dw': dw,
        'db': db
    }
    return grads


class LogisticRegression(object):
    """Logistic regression for binary classification

    Attributes:
        hyperparams - python dictionary defining model hyperparameters
                "training_iters": int, ideally multiple of 100
                "learning_rate": float32, scales parameter update rule
                "init_param_bound": float32, range in which to init weights
        w -- weight vector (n, 1) initialized within random range
        b -- bias initialized to zero, scalar
        cost_cache -- python list storing historical training error
        trained_params -- dictionary storing optimized params
        trained_grads -- dictionary storing optimized gradients

    Methods:
        ### ~WIP~ ###
        predict -- make binary predictions with trained model
        ### TO-DO ###
        accuracy -- provide accuracy metrics for the model
        plot_cost -- plot training error over number of training iterations
        #############
    """
    def __init__(self, hyperparams):
        self.hyperparams = hyperparams
        self.w, self.b = self.init_params
        self.cost_cache = []
        self.trained_params = {}
        self.trained_grads = {}

    def run_sgd(self, X_train, y_train, print_cost=True):
        """Fit model to training data with stochastic gradient descent

        Args:
            X_train -- train set: feature matrix (n, m_train)
            y_train -- train set: label vector (1, m_train)
            print=True -- print cost to console log

        Return:
            self
        """
        # Define helper variables: iterations, learning rate
        epochs = self.hyperparams['training_iters']
        alpha = self.hyperparams['learning_rate']
        # Training with gradient descent
        for i in range(epochs):
            # Forward and backward pass gives cost and gradients for each training iter
            y_hat =  hypothesis(self.w, self.b, X_train)
            cost = cross_entropy(y_hat, y_train)
            grads = compute_grads(y_hat, y_train, X_train)

            # Assertions
            assert(grads['dw'].shape == self.w.shape)
            assert(grads['db'].dtype == float)
            assert(cost.shape == ())

            # Update rule for tweaking parameters
            self.w = self.w - alpha * grads['dw']
            self.b = self.b - alpha * grads['db']
            # Record model error every 100 iterations
            if print_cost and i % 100 == 0:
                self.cost_cache.append(cost)
                print('Error after {} epochs: {}'.format(i + 1, cost))
        # Store optimized parameters
        self.trained_params = {
            'w': self.w,
            'b': self.b
        }
        # Store optimized gradients
        self.trained_grads = {
            'dw': grads['dw'],
            'db': grads['db']
        }
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
    def init_params(self):
        bound = self.hyperparams['init_param_bound']
        n = self.X_train.shape[0]
        w = np.random.randn(n, 1) * bound
        b = np.zeros((1, 1))
        return w, b
