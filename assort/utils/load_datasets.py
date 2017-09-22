import os
import numpy as np


def get_profit():
    # Load data
    fname = 'chain_profits.txt'
    fpath = os.path.join('examples', 'datasets', fname)
    data = np.loadtxt(fpath, delimiter=',')

    # Prepare data
    m, n = data.shape[0], 1
    X_train = data[:, 0].reshape((m, n))
    y_train = data[:, 1].reshape((m, 1))

    # Make up some test points
    X_test = np.array([3.5, 7])

    return (X_train, y_train, X_test)


def get_housing():
    # Load data
    fname = 'house_prices.txt'
    fpath = os.path.join('examples', 'datasets', fname)
    data = np.loadtxt(fpath, delimiter=',')

    # Prepare data
    m, n = data.shape[0], data.shape[1] - 1
    X_train = data[:, 0:2].reshape((m, n))
    y_train = data[:, -1].reshape((m, 1))

    # Test, dummy data
    X_test = np.array([[2500, 3], [1000, 2], [1400, 3]])
    return (X_train, y_train, X_test)
