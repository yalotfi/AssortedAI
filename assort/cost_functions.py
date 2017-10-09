import numpy as np


class MeanSquaredError(object):
    """docstring for MeanSquaredError"""

    def __init__(self, labels, preds, examples=None):
        try:
            assert(labels.shape == preds.shape)
        except ValueError:
            print("Actual and predicted label dimensions must match!")
        self.m = labels.shape[0]
        self.y_true = labels
        self.y_hat = preds
        self.loss = self.y_hat - self.y_true
        self.m = self.y_hat.shape[0]
        self.X = examples

    @property
    def get_cost(self):
        return (1 / (2 * self.m)) * np.sum(np.square(self.loss))

    @property
    def get_grad(self):
        try:
            assert(self.X is not None)
        except ValueError:
            print("Must provide feature matrix, X, to compute derivative!")
        return (1 / self.m) * np.dot(self.X.T, self.loss)


def mse(y_hat, y_train, derivative=False, X=None):
    """Mean Squared Error

    Arguments:
        y_hat -- predicted labels - vector with shape (m, 1)
        y_train -- actual labels -  vector with shape (m, 1)
        derivative -- if true, returns the gradient of the cost function given
                      the feature matrix
        X -- feature matrix, X, with shape (m, n)

    Returns:
        If derivative -- gradient of MSE cost function
        Otherwise -- MSE cost between prediction labels and actual labels

    Raises:
        ValueError: If derivative and X is none b/c X needed to compute
                    the gradient
    """
    try:
        assert(y_hat.shape[0] == y_train.shape[0])
        assert(y_hat.shape == y_train.shape)
    except ValueError:
        print("predicted and actual label vectors need same dims")

    m = y_train.shape[0]  # Number of training examples
    loss = y_hat - y_train
    if derivative:
        try:
            assert(X is not None)
        except ValueError:
            print("Need to provide feature matrix, X")
        # return (1 / m) * np.sum(np.dot(X.T, loss))
        return (1 / m) * np.dot(X.T, loss)
    else:
        return (1 / (2 * m)) * np.sum(np.square(loss))


def binary_cross_entropy(y_hat, y_train, derivative=False, X=None):
    """Categorical cross entropy

    Arguments:
        y_hat -- predicted labels - vector with shape (m, 1)
        y_train -- actual labels -  vector with shape (m, 1)
        derivative -- if true, returns the gradient of the cost function given
                      the feature matrix
        X -- feature matrix, X, with shape (m, n)

    Returns:
        If derivative -- Python dictionary of gradients w respect to weights
                         and biases as follows:
                            grads = {
                                "dw": dw,
                                "db": db
                            }
        Otherwise -- Log loss given prediction labels and actual labels

    Raises:
        ValueError: Need feature matrix, X, to compute the gradient
    """
    try:
        assert(y_hat.shape[0] == y_train.shape[0])
        assert(y_hat.shape == y_train.shape)
    except ValueError:
        print("predicted and actual label vectors need same dims")

    m = y_train.shape[0]  # Number of training examples
    if derivative:
        try:
            assert(X is not None)
        except ValueError:
            print("Need to provide feature matrix, X")
        dZ = y_hat - y_train
        dw = (1 / m) * np.dot(X.T, dZ)
        db = (1 / m) * np.sum(dZ)
        grads = {
            "dw": dw,
            "db": db
        }
        return grads
    else:
        return -(1 / m) * np.sum(y_train * np.log(y_hat) + (1 - y_train) * np.log(1 - y_hat))


def categorical_cross_entropy(y_hat, y_label, derivative=False, X=None):
    m = y_label.shape[1]
    if derivative:
        dZ = y_label - y_hat
        dw = -(1 / m) * np.dot(X, dZ.T)
        db = -(1 / m) * np.sum(dZ)
        grads = {
            "dw": dw,
            "db": db
        }
        return grads
    else:
        return -(1 / m) * np.sum(y_hat * np.log(y_label))
