import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.getcwd()))
from assort.datasets import get_mnist
from assort.preprocessing import rescale
from assort.linear import LogisticRegression


def main():
    (X_train, y_train), (X_test, y_test) = get_mnist(download=False,
                                                     serialize=False,
                                                     binary=True,
                                                     bin_digits=[0, 1],
                                                     flatten=True)

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    # Rescale pixel values
    X_train_norm = rescale(X_train)
    X_test_norm = rescale(X_test)

    # Build the model and evaluate
    model = LogisticRegression(epochs=400, lr=10e-5, lmda=10e-5)
    model.fit(X_train_norm, y_train)
    print(model.evaluate(X_train_norm, y_train))
    print(model.evaluate(X_test_norm, y_test))


if __name__ == '__main__':
    main()
