import numpy as np


def one_hot(labels, k_classes):
    """Encode each training label into a one-hot vector

    Arguments
    ---------
        labels : ndarray
            Column vector with shape (m, 1)
        k_classes : int
            Number of classes

    Returns
    -------
    ndarray
        encoded_labels as matrix with shape (m, k_classes)
    """
    m = labels.shape[0]
    k = k_classes + 1
    encoded_labels = np.zeros((m, k), dtype='float32')
    for i in range(m):
        labidx = int(labels[i][0])
        encoded_labels[i, labidx] = 1
    return encoded_labels
