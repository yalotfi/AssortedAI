import os
import sys
import time as t
import numpy as np

sys.path.insert(0, os.path.join(os.getcwd()))
from assort.utils.mnist_util import MNISTReader


def get_profit():
    # Load data
    fname = 'chain_profits.txt'
    fpath = os.path.join('assort', 'datasets', fname)
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
    fpath = os.path.join('assort', 'datasets', fname)
    data = np.loadtxt(fpath, delimiter=',')

    # Prepare data
    m, n = data.shape[0], data.shape[1] - 1
    X_train = data[:, 0:2].reshape((m, n))
    y_train = data[:, -1].reshape((m, 1))

    # Test, dummy data
    X_test = np.array([[2500, 3], [1000, 2], [1400, 3]])
    return (X_train, y_train, X_test)


def get_mnist(download=True, serialize=False, binary=False, flatten=False):
    """Retrieve MNIST datasets and return tuples of train/test sets"""

    # Load MNIST data
    directory = os.path.join('assort', 'datasets', 'mnist')
    reader = MNISTReader(directory, download)
    (X_train, y_train) = reader.train_set
    (X_test, y_test) = reader.test_set

    m_train, m_test = X_train.shape[0], X_test[0]
    px_row, px_col = X_train.shape[1], X_train.shape[2]

    if serialize:
        fpath = os.path.join(directory, 'mnist.npz')
        np.savez(fpath,
                 X_train=X_train, y_train=y_train,
                 X_test=X_test, y_test=y_test)

    if binary:
        train_zero_idxs = np.where(y_train == 0)[0]
        print(train_zero_idxs.shape)
        print(X_train[train_zero_idxs, :, :].shape)
        # print(np.take(X_train, train_zero_idxs).shape)
        # (train_ones_idxs, train_ones_labs) = np.where(y_train == 1)
        # (test_zero_idxs, test_zero_labs) = np.where(y_test == 0)
        # (test_ones_idxs, test_ones_labs) = np.where(y_test == 1)
        # X_train_bin = np.take(X_train, )


if __name__ == '__main__':
    get_mnist(download=False, serialize=False, binary=True, flatten=False)
