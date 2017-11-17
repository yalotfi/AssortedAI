import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.getcwd()))
from assort.utils.load_datasets import get_mnist
from assort.preprocessing import norm
from assort.preprocessing import one_hot as ova
from assort.regression.logistic import SoftmaxRegression


def main():
    # Load the MNIST dataset
    (X_train, y_train), (X_test, y_test) = get_mnist(download=False,
                                                     serialize=False,
                                                     binary=False,
                                                     bin_digits=[0, 1],
                                                     flatten=True)

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    # Perform normalization
    X_train_norm = norm.rescale(X_train)
    X_test_norm = norm.rescale(X_test)
    # X_train_norm = norm.standardize(X_train)
    # X_test_norm = norm.standardize(X_test)
    print(np.mean(X_train_norm))
    print(np.std(X_train_norm))
    print(np.max(X_train_norm))
    print(np.min(X_train_norm))

    # Create one-hot encoded label sets:
    digit_classes = int(np.max(y_train))
    y_train_ova = ova.one_hot_encode(y_train, digit_classes)
    y_test_ova = ova.one_hot_encode(y_test, digit_classes)

    model = SoftmaxRegression(epochs=1000, lr=0.005, lmda=0.05)
    model.fit(X_train_norm, y_train_ova)
    print(model.evaluate(X_train_norm, y_train_ova))
    print(model.evaluate(X_test_norm, y_test_ova))
    y_pred = model.predict(X_test_norm)
    # model.test_train(X, Y)


if __name__ == '__main__':
    main()
