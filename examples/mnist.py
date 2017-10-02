import sys
import os
import numpy as np
import time as t

sys.path.insert(0, os.path.join(os.getcwd()))
from assort.utils.load_datasets import get_mnist
# from assort.utils.preprocessing import one_hot_encode
# from assort.utils.preprocessing import normalize


def main():
    tic = t.time()
    (X_train, y_train), (X_test, y_test) = get_mnist(download=False,
                                                     serialize=False,
                                                     binary=False,
                                                     bin_digits=[0, 1],
                                                     flatten=True)
    toc = t.time() - tic
    print("Preprocessing Time: {}\n".format(toc))
    print("Train Set:\nfeatures: {} | labels: {}\n".format(
        X_train.shape, y_train.shape))
    print("Test Set:\nfeatures: {} | labels: {}\n".format(
        X_test.shape, y_test.shape))


if __name__ == '__main__':
    main()
