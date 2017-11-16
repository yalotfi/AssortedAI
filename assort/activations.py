import numpy as np


def sigmoid(z):
    """Sigmoid or logistic activation function"""
    return 1 / (1 + np.exp(-z))


# def softmax(z):
#     """Softmax activation function (generalized sigmoid)"""
#     z -= np.max(z)
#     return np.exp(z) / np.sum(np.exp(z), axis=0)

def softmax(arr, axis=None):
    X = np.atleast_2d(arr)
    if axis is None:
        axis = next(j[0] for j in enumerate(X.shape) if j[1] > 1)
    X = X - np.expand_dims(np.max(X, axis=axis), axis)
    X = np.exp(X)
    probs = X / np.expand_dims(np.sum(X, axis=axis), axis)
    if len(arr.shape) == 1:
        return probs.flatten()
    else:
        return probs


def tanh(z):
    """Hyperbolic tangent activation function"""
    sinh = np.exp(z) - np.exp(-z)
    cosh = np.exp(z) + np.exp(-z)
    return sinh / cosh


def relu(z):
    """Rectified Linear Unit activation function"""
    return np.maximum(z, 0, z)


def leaky_relu(z, a=0.01):
    """Leaky Rectified Linear Unit activation function"""
    return np.maximum(z, a * z, z)
