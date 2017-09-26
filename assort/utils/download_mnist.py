import os
import gzip
import struct
import numpy as np

from urllib import request


def download_mnist(url, fname, directory):
    """Download the file from the given url, filename, and directory."""
    if not os.path.exists(directory):
        print("Creating directory %s" % directory)
        os.mkdir(directory)
    else:
        print("Directory exists: %s" % directory)
    filepath = os.path.join(directory, fname)
    if not os.path.exists(filepath):
        print("Downloading %s to %s" % (fname, filepath))
        local_fname, _ = request.urlretrieve(url + fname, filepath)
        statinfo = os.stat(filepath)
        print("Successfully downloaded %s bytes %s" % (fname, statinfo.st_size))
    else:
        print("File %s exists in %s" % (fname, filepath))
    return filepath


def read_mnist(fimg, flab):
    """Read raw MNIST data files and load them into NumPy arrays."""
    with gzip.open(flab) as f:
        magic, num = struct.unpack(">II", f.read(8))
        label = np.fromstring(f.read(), dtype=np.int8)

    m = label.shape[0]
    with gzip.open(fimg, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        image = np.fromstring(f.read(), dtype=np.uint8).reshape(m, rows, cols)
    return (image, label)


def main():
    url='http://yann.lecun.com/exdb/mnist/'
    fnames = ['train-images-idx3-ubyte.gz',
              'train-labels-idx1-ubyte.gz',
              't10k-images-idx3-ubyte.gz',
              't10k-labels-idx1-ubyte.gz']
    directory = os.path.join('assort', 'datasets', 'mnist')
    # Testing
    for fname in fnames:
        filepath = download_mnist(url, fname, directory)

    X_train, y_train = read_mnist(fnames[0], fnames[1])
    print(X_train.shape)
    print(y_train.shape)
    X_test, y_test = read_mnist(fnames[2], fnames[3])
    print(X_test.shape)
    print(y_test.shape)


if __name__ == '__main__':
    main()
