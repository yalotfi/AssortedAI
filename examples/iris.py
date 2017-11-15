import sys
import os
import pprint as pp
import numpy as np


sys.path.insert(0, os.path.join(os.getcwd()))
from assort.utils import load_datasets
from assort.regression.logistic import SoftmaxRegression
# from assort.regression.softmax import SoftmaxRegression


def main():
    (X_train, y_train), (X_test, y_test) = load_datasets.get_iris(99)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    model = SoftmaxRegression(epochs=5000, lr=0.05, lmda=0.05)
    model.fit(X_train, y_train)
    # model.plot_error()


if __name__ == '__main__':
    main()
