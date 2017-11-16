import sys
import os
import numpy as np


sys.path.insert(0, os.path.join(os.getcwd()))
from assort.utils import load_datasets
from assort.preprocessing import one_hot as ova
from assort.regression.logistic import SoftmaxRegression


def main():
    # Load Iris Dataset
    (X_train, y_train), (X_test, y_test) = load_datasets.get_iris(99)

    # One-Hot Encoding
    iris_classes = int(np.max(y_train))
    y_train_ova = ova.one_hot_encode(y_train, iris_classes)
    y_test_ova = ova.one_hot_encode(y_test, iris_classes)

    print(X_train.shape)
    print(y_train_ova.shape)
    print(X_test.shape)
    print(y_test_ova.shape)

    model = SoftmaxRegression(epochs=1000, lr=0.005, lmda=0.01)
    model.fit(X_train, y_train_ova)
    print(model.evaluate(X_train, y_train))
    print(model.evaluate(X_test, y_test))
    # model.plot_error()


if __name__ == '__main__':
    main()
