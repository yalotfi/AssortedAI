import numpy as np


def rescale(X):
    """Rescale data between [0, 1]"""
    return (X - np.min(X)) / (np.max(X) - np.min(X))


def standardize(X):
    """Standardize features to have zero mean and unit-variance"""
    # X - X_bar / variance
    pass


def one_hot_encode(labels, n_classes):
    """Encode a label vector into a one-hot matrix

    Arguments:
        labels - column vector with shape =
    """
    m_examples = labels.shape[0]  # m, training examples
    encoded_labels = np.zeros((m_examples, n_classes), dtype='float32')
    for i in range(m_examples):
        encoded_labels[i, labels[i]] = 1
    return encoded_labels
