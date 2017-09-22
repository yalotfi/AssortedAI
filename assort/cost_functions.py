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


def cross_entropy(y_hat, y_train, derivative=False, X=None):
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
