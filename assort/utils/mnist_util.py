import sys
import os
import gzip
import struct
import numpy as np

from assort.utils.download_util import download


class MNISTReader(object):
    """docstring for MNISTReader."""
    def __init__(self, directory, download_flag=False):
        super(MNISTReader, self).__init__()
        self.directory = directory
        self.download_flag = download_flag
        self.fnames = {
            "train_set": {
                "features": 'train-images-idx3-ubyte.gz',
                "labels": 'train-labels-idx1-ubyte.gz'
                },
            "test_set": {
                "features": 't10k-images-idx3-ubyte.gz',
                "labels": 't10k-labels-idx1-ubyte.gz'
                }
            }
        self.train_set = self._load_train_set
        self.test_set = self._load_test_set

    def _download_mnist(self, which_set):
        """Download train or test MNIST sets from Yann LeCunn's public repo,
           dumping the gzip files in the given directory.
        """
        url = 'http://yann.lecun.com/exdb/mnist/'
        for key, fname in which_set.items():
            filepath = download(url, fname, self.directory)

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
        assert(rows == cols)  # 28 x 28
        return images, rows, cols

    def _read_mnist(self, which_set):
        """Try to download raw MNIST and return (feature, label) set."""
        img_path = os.path.join(self.directory, which_set["features"])
        lab_path = os.path.join(self.directory, which_set["labels"])
        if self.download_flag:
            self._download_mnist(which_set)
        labels, m = self._read_labs(lab_path)
        images, r, c = self._read_imgs(img_path)
        return (images.reshape(m, r, c), labels.reshape(m, 1))

    @property
    def _load_train_set(self):
        return self._read_mnist(which_set=self.fnames["train_set"])

    @property
    def _load_test_set(self):
        return self._read_mnist(which_set=self.fnames["test_set"])
