import numpy as np

from utils.activations import sigmoid
from utils.cost_functions import log_loss


class SoftmaxRegression(object):
    """Generalized logistic regression for K classes

    Attributes:
        hyperparams - python dictionary defining model hyperparameters
                "training_iters": int, ideally multiple of 100
                "learning_rate": float32, scales parameter update rule
                "init_param_bound": float32, range in which to init weights
        n -- feature size, scalar int
        m -- training examples, scalar int
        k -- number of classes, scalar int
        X -- feature matrix (n, m), np.ndarray
        Y -- label matrix (k, m), np.ndarray
        w -- weight vector (n, k), np.ndarray
        b -- bias initialized to zero, scalar type float
        cost_cache -- python list storing historical training error
        trained_params -- dictionary storing optimized params
        trained_grads -- dictionary storing optimized gradients

    Methods:
        ### ~WIP~ ###
        predict -- make binary or multiclass predictions with trained model
        ### TO-DO ###
        accuracy -- provide accuracy metrics for the model
        plot_cost -- plot training error over number of training iterations
        #############
    """
    def __init__(self, X, Y, hyperparams):
        self.hyperparams = hyperparams
        self.n = X.shape[0]
        self.m = X.shape[1]
        self.k = Y.shape[0]
        self.X = X
        self.Y = Y
        self.w, self.b = self.init_params
        self.cost_cache = []
        self.trained_params = {}
        self.trained_grads = {}

    def optimize(self, persist=True, print_cost=True):
        """Fit softmax regression to training data

        Args:
            persist=True -- save to disk or not, bool
            print=True -- print cost to console log

        Return:
            self

        Description:
            Requires the training iterations for SGD and learning rate which
            are both used to update and train the model parameters

            SoftmaxClassifier.optimize( ... ) will update the class object's trained
            gradients and parameter attributes as dictionaries so that they can
            be accessed later.
        """
        # Define helper variables: iterations, learning rate
        epochs = self.hyperparams['training_iters']
        alpha = self.hyperparams['learning_rate']
        # Training with gradient descent
        for i in range(epochs):
            # Forward and backward pass gives cost and gradients for i-th epoch
            grads, cost = self._propagate()
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

    def predict(self, X, binary=True, binary_threshold=0.5):
        """Predict labels on a given data set

        Arguments:
            X -- feature matrix with shape (feature_size, examples), numpy ndarray
            binary -- binary or multiclass classification, bool
            binary_threshold -- probability threshold to pick binary class
        """
        preds = np.zeros((self.k, self.m))
        w = self.trained_params['w']
        b = self.trained_params['b']
        y_hat = sigmoid(np.dot(w.T, X) + b)
        # If binary classification:
        if binary:
            # Take probability vector, y_hat: output predicted classes
            for i in range(y_hat.shape[0]):
                if y_hat[0, i] <= binary_threshold:
                    preds[0, i] = 0
                elif y_hat[0, i] > binary_threshold:
                    preds[0, i] = 1
            return preds
        # Otherwise, multiclass classification
        else:
            preds = np.amax(y_hat, axis=1)
            return preds

    def _propagate(self):
        # Step 1: Forward Propagation
        Y_hat = sigmoid(np.dot(self.w.T, self.X) + self.b)
        cost = np.squeeze(log_loss(Y_hat, self.Y))

        # Step 2: Backward Propagation
        dZ = Y_hat - self.Y
        dw = (1 / self.m) * np.dot(self.X, dZ.T)
        db = (1 / self.m) * np.sum(dZ)

        # Save gradients in a dictionary
        grads = {
            'dw': dw,
            'db': db
        }
        # Assert that dw and w have same dims, db is float, and cost squeezed
        assert(dw.shape == self.w.shape)
        assert(db.dtype == float)
        assert(cost.shape == ())
        # Return gradients and cost
        return grads, cost

    @property
    def init_params(self):
        bound = self.hyperparams['init_param_bound']
        w = np.random.randn(self.n, self.k) * bound
        b = np.zeros((1, 1))
        return w, b
