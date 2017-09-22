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
