import os
import gzip
import struct
import re
import numpy as np

from urllib import request


def download_mnist(url, fname, directory):
    """Download the file from the given url, filename, and directory."""
    if not os.path.exists(directory):
        print("Creating directory %s" % directory)
        os.mkdir(directory)
    filepath = os.path.join(directory, fname)
    if not os.path.exists(filepath):
        print("Downloading %s to %s" % (url, filepath))
        filepath, _ = request.urlretrieve(url, filepath)
        statinfo = os.stat(filepath)
        print("Successfully downloaded", fname, statinfo.st_size, "bytes")
    return filepath


def read_labels(flab):
    """Read raw MNIST data files and load them into NumPy arrays."""
    with gzip.open(flab) as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromstring(f.read(), dtype=np.int8)
    return labels


def read_mnist(fimg, flab):
    """Read raw MNIST data files and load them into NumPy arrays."""
    with gzip.open(flab) as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromstring(f.read(), dtype=np.int8)
    m = len(labels)
    with gzip.open(fimg, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromstring(f.read(), dtype=np.uint8)
    return (images.reshape(m, rows, cols), labels.reshape(m, 1))


def main():
    # directory = os.path.join('assort', 'datasets', 'mnist')
    url = 'http://yann.lecun.com/exdb/mnist/'
    fnames = ['train-images-idx3-ubyte.gz',
              'train-labels-idx1-ubyte.gz',
              't10k-images-idx3-ubyte.gz',
              't10k-labels-idx1-ubyte.gz']
    for fname in fnames:
        request.urlretrieve(url, fname)
    # for fname in fnames:
    #     if not os.path.exists(directory + fname):
    #         fpath = download_mnist(url, fname, directory)
    #         print("File in: ", fpath)
    #     else:
    #         print("File %s exists in %s" % (fname, directory))
    # print(os.path.join(directory, fnames[1]))
    # y_train = read_labels(os.path.join(directory, fnames[1]))
    # img_dir = os.path.join(directory, fnames[0])
    # lab_dir = os.path.join(directory, fnames[1])
    (X_train, y_train) = read_mnist(fnames[0], fnames[1])
    print(X_train.shape)
    print(y_train.shape)


if __name__ == '__main__':
    main()
