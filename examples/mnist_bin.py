import sys
import os
import numpy as np
import time as t

sys.path.insert(0, os.path.join(os.getcwd()))
from assort.utils.load_datasets import get_mnist
from assort.regression.logistic import LogisticRegression


def main():
    tic = t.time()
    (X_train, y_train), (X_test, y_test) = get_mnist(download=False,
                                                     serialize=False,
                                                     binary=True,
                                                     bin_digits=[0, 1],
                                                     flatten=True)
    toc = t.time() - tic
    print("Preprocessing Time: {}\n".format(toc))
    print("Train Set:\nfeatures: {} | labels: {}\n".format(
        X_train.shape, y_train.shape))
    print("Test Set:\nfeatures: {} | labels: {}\n".format(
        X_test.shape, y_test.shape))

    # Perform normalization
    X_train_norm, X_test_norm = X_train / 255, X_test / 255
    print("Train set min: {} | max: {}".format(
        np.min(X_train_norm), np.max(X_train_norm)))

    # Define model hyperparameters
    hyperparameters = {
        "learning_rate": 0.005,
        "training_iters": 2000,
        "init_param_bound": 0.01
    }

    # Train Logistic Regression model
    model = LogisticRegression(hyperparameters)
    model = model.gradient_descent(X_train_norm, y_train)


if __name__ == '__main__':
    main()
