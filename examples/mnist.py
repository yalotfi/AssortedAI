import sys
import os

sys.path.insert(0, os.path.join(os.getcwd()))
from assort.utils.load_datasets import get_mnist
from assort.preprocessing.one_hot import one_hot_encode
from assort.regression.softmax import SoftmaxRegression


def main():
    # Load the MNIST dataset
    (X_train, y_train), (X_test, y_test) = get_mnist(download=False,
                                                     serialize=False,
                                                     binary=False,
                                                     bin_digits=[0, 1],
                                                     flatten=True)

    # Perform normalization
    X_train_norm, X_test_norm = X_train / 255, X_test / 255

    # Create one-hot encoded label sets:
    y_train_k = one_hot_encode(y_train, 10)
    y_test_k = one_hot_encode(y_test, 10)

    X, Y = X_train_norm.T, y_train_k.T
    print(X.shape)
    print(Y.shape)
    model = SoftmaxRegression()
    model = model.gradient_descent(X, Y)
    model.plot_error()
    # model.test_train(X, Y)


if __name__ == '__main__':
    main()
