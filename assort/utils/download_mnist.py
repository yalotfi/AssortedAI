import numpy as np
import os
import gzip
import struct

from urllib import request

def download_mnist(url, force_download=True):
    """Download the file from the given url and return the file name."""
    fname = url.split("/")[-1]
    if force_download or not os.path.exists(fname):
        request.urlretrieve(url, fname)
    return fname


def read_mnist_imgs(fname):
    """Read and unpack zipped mnist images"""
    with open(fname, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        image = np.fromstring(f.read(), dtype=np.int8)
    return image


def read_mnist_labs(fname):
    """Read and unpack zipped mnist labels"""
    with open(fname) as f:
        magic, num = struct.unpack(">II", f.read(8))
        label = np.fromstring(f.read(), dtype=np.int8).reshape(len(label), 1)


def read_data(label_url, image_url):
    '''
        Download labels (from the label_url) and images
        (from the image_url). Load them into Numpy arrays (label and images).
        Example: label[0] corresponds to label of the image at image[0].
    '''
    with gzip.open(download_data(label_url)) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8).reshape(len(label), 1)

    with gzip.open(download_data(image_url), 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return (label, image)


def main():
    path='http://yann.lecun.com/exdb/mnist/'
    # (y_train, X_train) = read_data(
    #     path+'train-labels-idx1-ubyte.gz', path+'train-images-idx3-ubyte.gz')
    # (y_test, X_test) = read_data(
    #     path+'t10k-labels-idx1-ubyte.gz', path+'t10k-images-idx3-ubyte.gz')
    # print(X_train.shape, y_test.shape)
    fname = 'train-images-idx3-ubyte.gz'
    X_train = read_mnist_imgs(fname)
    print(X_train.shape)


if __name__ == '__main__':
    main()
