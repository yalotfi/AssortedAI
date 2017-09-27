import os
import gzip
import struct
import time as t
import numpy as np

from urllib import request

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


def get_mnist(directory, binary=False, flatten=True):
    directory = os.path.join('assort', 'datasets', 'mnist')
    reader = MNISTReader(directory)
    (X_train, y_train) = reader.train_set
    (X_test, y_test) = reader.test_set

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)


if __name__ == '__main__':
    url = 'http://yann.lecun.com/exdb/mnist/'
    train_img =  'train-images-idx3-ubyte.gz'
    print(url + train_img)

    # tic = t.time()
    # X_train = get_mnist()
    # print(t.time() - tic)
    # print(X_train.shape)
