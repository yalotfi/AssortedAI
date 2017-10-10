import numpy as np

from assort import _INITIALIZER_CONFIG
from assort.activations import sigmoid
from assort.cost_functions import BinaryCrossEntropy


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
                 X_train,
                 y_train,
                 weight_initializer='random_normal',
                 bias_initializer='zeros'):
        # Get training features and labels
        self.X_train = X_train
        self.y_train = y_train

        # Set paramter initializers
        if weight_initializer in _INITIALIZER_CONFIG:
            self.weight_initializer = _INITIALIZER_CONFIG[weight_initializer]()
        if bias_initializer in _INITIALIZER_CONFIG:
            self.bias_initializer = _INITIALIZER_CONFIG[bias_initializer]()

        # Assigned when model is training
        self.cost_cache = []
        self.trained_params = {}
        self.trained_grads = {}

    def gradient_descent(self, alpah, epochs, print_cost_freq=100):
        """Fit model to training data with stochastic gradient descent

        Arguments:
            alpha -- Learning rate
            epochs -- Training iterations
            print_cost_freq -- how often to print cost

        Return:
            self
        """
        # Define helper variables: iters, learning rate, dim, params
        m, n = self.X_train.shape[0], self.X_train.shape[1]
        k_classes = y_train.shape[1]
        w = self.weight_initializer((n, 1))
        b = self.bias_initializer((1, k_classes))

        # Training with gradient descent
        print("Training model...")
        for i in range(epochs):
            # 1) Make prediction
            y_hat = sigmoid(np.dot(X, w) + b)
            # 2) Compute loss and gradients of parameters
            bce = BinaryCrossEntropy(y_train, y_hat, X_train)
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
