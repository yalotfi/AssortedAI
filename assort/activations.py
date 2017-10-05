import numpy as np


def sigmoid(z):
    """Sigmoid or logistic activation function"""
    return 1 / (1 + np.exp(-z))


def softmax(z, axis=1):
    """Softmax activation function (generalized sigmoid)"""
    z -= np.max(z)
    return np.exp(z) / np.sum(np.exp(z))


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
