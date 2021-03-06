import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.getcwd()))
from assort.utils.load_datasets import get_mnist
from assort.preprocessing import feature_scaling as norm
from assort.preprocessing import one_hot as ova
from assort.linear.logistic import SoftmaxRegression


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

    # Rescale pixel values
    X_train_norm = norm.rescale(X_train)
    X_test_norm = norm.rescale(X_test)

    # Create one-hot encoded labels:
    digit_classes = int(np.max(y_train))
    y_train_ova = ova.one_hot_encode(y_train, digit_classes)
    y_test_ova = ova.one_hot_encode(y_test, digit_classes)

    # Build the model and evaluate
    model = SoftmaxRegression(epochs=250, lr=10e-5, lmda=10e-5)
    model.fit(X_train_norm, y_train_ova)
    print(model.evaluate(X_train_norm, y_train))
    print(model.evaluate(X_test_norm, y_test))


if __name__ == '__main__':
    main()
