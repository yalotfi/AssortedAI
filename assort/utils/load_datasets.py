import os
import gzip
import struct
import time as t
import numpy as np

from urllib import request


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


def get_mnist(binary=False):
    # Download file from Yann LeCunn's MNIST repo
    url = 'http://yann.lecun.com/exdb/mnist/'
    train_img =  'train-images-idx3-ubyte.gz'
    train_lab = 'train-labels-idx1-ubyte.gz'
    test_img = 't10k-images-idx3-ubyte.gz'
    test_lab = 't10k-labels-idx1-ubyte.gz'

    # Download and unpack training images into X_train feature matrix
    with gzip.open(request.urlretrieve(url + train_img, train_img)) as trainimg:
        magic, num = struct.unpack('>II', trainimg.read(8))
        X_train = np.fromstring(flbl.read(), dtype=np.float32)

    # # Download and unpack training images into X_test feature matrix
    # with gzip.open(urllib.urlretrieve(url + test_img, test_img)) as testimg:
    #     magic, num = struct.unpack('>II', flbl.read(8))
    #     X_test = np.fromstring(flbl.read(), dtype=np.float32)
    #
    # # Download and unpack training labels into y_train label vector
    # with gzip.open(urllib.urlretrieve(url + test_img, test_img)) as fimg:
    #     magic, num = struct.unpack('>II', flbl.read(8))
    #     X_test = np.fromstring(flbl.read(), dtype=np.float32)
    #
    # # Download and unpack testing labels into y_test label vector
    # with gzip.open(urllib.urlretrieve(url + test_img, test_img)) as flbl:
    #     magic, num = struct.unpack('>II', flbl.read(8))
    #     X_test = np.fromstring(flbl.read(), dtype=np.float32)

    return X_train


if __name__ == '__main__':
    url = 'http://yann.lecun.com/exdb/mnist/'
    train_img =  'train-images-idx3-ubyte.gz'
    print(url + train_img)

    # tic = t.time()
    # X_train = get_mnist()
    # print(t.time() - tic)
    # print(X_train.shape)
