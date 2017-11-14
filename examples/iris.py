import sys
import os
import pprint as pp
import numpy as np


sys.path.insert(0, os.path.join(os.getcwd()))
from assort.utils import load_datasets
from assort.preprocessing import sampling
from assort.regression.softmax import SoftmaxRegression


def main():
    (X_train, y_train), (X_test, y_test) = load_datasets.get_iris(99)
    print(X_train.T.shape)
    print(y_train.T.shape)
    print(X_test.shape)
    print(y_test.shape)

    model = SoftmaxRegression()
    model.gradient_descent(X_train.T, y_train.T)
    model.plot_error()


if __name__ == '__main__':
    main()
