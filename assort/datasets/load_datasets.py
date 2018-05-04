import os
import csv
import numpy as np

from . import download
from . import MNISTReader
from ..preprocessing import train_test_split


DATADIR = os.path.abspath(os.path.dirname(__file__))


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
    fpath = os.path.join(DATADIR, 'housing', fname)
    data = np.loadtxt(fpath, delimiter=',')

    # Prepare data
    m, n = data.shape[0], data.shape[1] - 1
    X_train = data[:, 0:2].reshape((m, n))
    y_train = data[:, -1].reshape((m, 1))

    # Test, dummy data
    X_test = np.array([[2500, 3], [1000, 2], [1400, 3]])
    return (X_train, y_train, X_test)


def get_spam(seed, test_size=0.3):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/'
    fname = "spambase.data"
    directory = os.path.join(DATADIR, 'spam')
    filepath = download(url, fname, directory)
    with open(filepath, 'r') as f:
        emails = np.asarray(
            [row for row in csv.reader(f, delimiter=',')], dtype=np.float32)
    np_X = emails[:, :-1]
    np_y = emails[:, -1].reshape(np_X.shape[0], 1)
    return train_test_split(np_X, np_y, seed=seed, test_size=test_size)


def get_iris(seed, test_size=0.3):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/'
    fname = "iris.data"
    directory = os.path.join(DATADIR, 'iris')
    filepath = download(url, fname, directory)
    classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    X = []
    y = []
    with open(filepath, 'r') as f:
        csvreader = csv.reader(f, delimiter=',')
        for row in csvreader:
            if len(row) == 5:
                X.append(row[:-1])
                y.append(classes.index(row[-1]))
            else:  # Last empty row can just be ignored
                pass
    np_X = np.asarray(X, dtype=np.float32)
    np_y = np.asarray(y, dtype=np.float32).reshape((len(y), 1))
    return train_test_split(np_X, np_y, seed=seed, test_size=test_size)


def get_mnist(download=True, serialize=False,
              binary=False, bin_digits=[0, 1], flatten=False):
    """Retrieve MNIST datasets and return tuples of train/test sets"""

    # Load MNIST data - supply download and flatten flags to MNIST Reader
    directory = os.path.join(DATADIR, 'mnist')
    reader = MNISTReader(directory, download, flatten)

    # Pull the train and test sets stored as attributes of the reader
    (X_train, y_train) = reader.train_set
    (X_test, y_test) = reader.test_set

    # Flag - serialize arrays to default directory: assort\datasets\mnist\
    if serialize:
        fpath = os.path.join(directory, 'mnist.npz')
        np.savez(fpath,
                 X_train=X_train, y_train=y_train,
                 X_test=X_test, y_test=y_test)
        print("Saved MNIST arrays to disk here: {}\n".format(fpath))

    # Flag - subset for binary classification
    if binary:
        # Subset by logical indexing
        a, b = bin_digits[0], bin_digits[1]
        train_digits = np.where(np.logical_or(y_train == a, y_train == b))
        test_digits = np.where(np.logical_or(y_test == a, y_test == b))
        X_train_bin = X_train[train_digits[0]]  # (m_bin_train, 28, 28)
        y_train_bin = y_train[train_digits[0]]  # (m_bin_train, 1)
        X_test_bin = X_test[test_digits[0]]  # (m_bin_test, 28, 28)
        y_test_bin = y_test[test_digits[0]]  # (m_bin_test, 1)
        return (X_train_bin, y_train_bin), (X_test_bin, y_test_bin)
    # Otherwise return full dataset of all 10-classes
    else:
        return (X_train, y_train), (X_test, y_test)
