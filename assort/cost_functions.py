import numpy as np


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
    assert(y_hat.shape[0] == y_train.shape[0])
    assert(y_hat.shape == y_train.shape)

    m = y_train.shape[0]  # Number of training examples
    loss = y_hat - y_train
    if derivative:
        try:
            assert(X is not None)
        except ValueError:
            print("Need to provide feature matrix, X")
        return (1 / m) * np.sum(np.dot(X.T, loss))
    else:
        return (1 / (2 * m)) * np.sum(np.square(loss))


def cross_entropy(y_hat, y):
    """Categorical cross entropy"""
    assert(y_hat.shape == y.shape)
    assert(y_hat.shape[0] == y.shape[0])
    m = y.shape[0]  # Number of training examples
    loss = -(1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    return loss
