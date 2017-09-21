import numpy as np

def sigmoid(z):
    """Sigmoid or logistic activation function"""
    return 1 / (1 + np.exp(-z))
