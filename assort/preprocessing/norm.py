import numpy as np


def rescale(X):
    """Rescale data between [0, 1]"""
    numer = X - np.min(X, axis=0)
    denom = np.max(X, axis=0) - np.min(X, axis=0)
    return numer / denom


def standardize(X):
    """Standardize features to have zero mean and unit-variance"""
    # X - X_bar / std
    x_bar = np.mean(X, axis=0)  # feature means
    sigma = np.std(X, axis=0)  # feature st deviations
    return (X - x_bar) / sigma
