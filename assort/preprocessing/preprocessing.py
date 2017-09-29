import numpy as np


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
