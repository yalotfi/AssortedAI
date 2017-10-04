import numpy as np


def one_hot_encode(labels, k_classes):
    """Encode each training label into a one-hot vector

    Arguments:
        labels - column vector with shape (m, 1)
        k_classes - classes to encode

    Return:
        encoded_labels - matrix with shape (m, k_classes)
    """
    m_examples = labels.shape[0]  # m, training examples
    encoded_labels = np.zeros((m_examples, k_classes), dtype='float32')
    for i in range(m_examples):
        encoded_labels[i, labels[i]] = 1
    return encoded_labels
