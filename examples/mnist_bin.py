import sys
import os

sys.path.insert(0, os.path.join(os.getcwd()))
from assort.utils.load_datasets import get_mnist
from assort.regression.logistic import LogisticRegression


def main():
    (X_train, y_train), (X_test, y_test) = get_mnist(download=False,
                                                     serialize=False,
                                                     binary=True,
                                                     bin_digits=[0, 1],
                                                     flatten=True)
    print("Train Set:\nfeatures: {} | labels: {}\n".format(
        X_train.shape, y_train.shape))
    print("Test Set:\nfeatures: {} | labels: {}\n".format(
        X_test.shape, y_test.shape))

    # Perform normalization
    X_train_norm, X_test_norm = X_train / 255, X_test / 255

    # Train Logistic Regression model
    model = LogisticRegression()
    model.gradient_descent(X_train_norm, y_train, alpha=0.0005, epochs=100)


if __name__ == '__main__':
    main()
