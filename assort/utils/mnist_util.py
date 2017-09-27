import sys
import os
import gzip
import struct
import numpy as np

sys.path.insert(0, os.path.join(os.getcwd()))
from assort.utils.download_util import download


class MNISTReader(object):
    """docstring for MNISTReader."""
    def __init__(self, directory, download_flag=False):
        super(MNISTReader, self).__init__()
        self.directory = directory
        self.download_flag = download_flag
        self.fnames = {"X_train": 'train-images-idx3-ubyte.gz',
                       "y_train": 'train-labels-idx1-ubyte.gz',
                       "X_test": 't10k-images-idx3-ubyte.gz',
                       "y_test": 't10k-labels-idx1-ubyte.gz'}
        self.train_set = self._load_train_set
        self.test_set = self._load_test_set

    def save_mnist(self):
        """Persist MNIST NumPy Arrays to disk"""
        pass

    def _download_mnist(self, directory):
        """Download raw MNIST files from Yann LeCunn's public repo and dumps
           the gzip files in the given directory.
        """
        url = 'http://yann.lecun.com/exdb/mnist/'
        for fname in self.fnames:
            filepath = download(url, fname, directory)

    def _read_labs(self, flab):
        """Unpack label file bytes into NumPy Array"""
        with gzip.open(flab) as f:
            magic, m = struct.unpack(">II", f.read(8))
            labels = np.fromstring(f.read(), dtype=np.int8)
        assert(m == labels.shape[0])
        return labels, m

    def _read_imgs(self, fimg):
        """Unpack image file bytes into NumPy Array"""
        with gzip.open(fimg, 'rb') as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.fromstring(f.read(), dtype=np.uint8)
        assert(rows == cols)
        img_dims = {"row": rows, "col": cols}
        return images, img_dims

    def _read_mnist(self, fimg, flab):
        """Choose to download raw MNIST and return feature/label sets."""
        img_path = os.path.join(self.directory, fimg)
        lab_path = os.path.join(self.directory, flab)
        if self.download_flag:
            self._download_mnist(self.directory)
        labels, m = self._read_labs(lab_path)
        images, dim = self._read_imgs(img_path)
        return (images.reshape(m, dim["row"], dim["col"]), labels.reshape(m, 1))

    @property
    def _load_train_set(self):
        return self._read_mnist(self.fnames["X_train"], self.fnames["y_train"])

    @property
    def _load_test_set(self):
        return self._read_mnist(self.fnames["X_test"], self.fnames["y_test"])


def main():
    directory = os.path.join('assort', 'datasets', 'mnist')
    reader = MNISTReader(directory)
    (X_train, y_train) = reader.train_set
    (X_test, y_test) = reader.test_set

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)


if __name__ == '__main__':
    main()
