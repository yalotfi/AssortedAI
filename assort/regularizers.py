import numpy as np


def l1_reg(w, lmda):
    """
    L1 Regularization Term

    Arguments
    ---------
    w : ndarray
        Model parameters to decay by regularization term
    lmda : float
        Regularization constant, lambda

    Returns
    -------
    float
        Regularization term to add to the cost
    """
    return lmda / 2 * np.sum(w)


def l2_reg(w, lmda, derivative=False):
    """
    L2 Regularization Term

    Arguments
    ---------
    w : ndarray
        Model parameters to decay by regularization term
    lmda : float
        Regularization constant, lambda
    derivative : bool
        Choose to return derivative of L2 term for backprop. Default is False

    Returns
    -------
    float
        Regularization term to add to the cost
    """
    if derivative:
        return lmda * w
    else:
        return lmda / 2 * np.sum(np.dot(w, w.T))
