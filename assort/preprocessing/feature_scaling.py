import numpy as np


def rescale(X):
    """Rescale data between [0, 1]"""
    numer = X - np.min(X)
    denom = np.max(X) - np.min(X)
    return numer / denom


def mean_normalize(X):
    """Normalize the mean of a given distribution"""
    numer = X - np.mean(X)
    denom = np.max(X) - np.min(X)
    return numer / denom


def standardize(X):
    """Standardize features to have zero mean and unit-variance"""
    # X - X_bar / std
    x_bar = np.mean(X)  # feature means
    sigma = np.std(X)  # feature st deviations
    return (X - x_bar) / sigma
