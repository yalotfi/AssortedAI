import sys
import os
import numpy as np
import time as t

sys.path.insert(0, os.path.join(os.getcwd()))
from assort.utils.load_datasets import get_mnist
from assort.regression.logistic import SoftmaxRegression
from assort.preprocessing.one_hot import one_hot_encode

from assort.activations import sigmoid


def main():
    tic = t.time()
    (X_train, y_train), (X_test, y_test) = get_mnist(download=False,
                                                     serialize=False,
                                                     binary=False,
                                                     bin_digits=[0, 1],
                                                     flatten=True)
    toc = t.time() - tic
    print("Preprocessing Time: {}\n".format(toc))

    # Perform normalization
    X_train_norm, X_test_norm = X_train / 255, X_test / 255
    print("Train set min: {} | max: {}".format(
        np.min(X_train_norm), np.max(X_train_norm)))

    # Create one-hot encoded label sets:
    y_train_k = one_hot_encode(y_train, 10)
    y_test_k = one_hot_encode(y_test, 10)

    print("Train Set:\nfeatures: {} | labels: {}\n".format(
        X_train_norm.shape, y_train_k.shape))
    print("Test Set:\nfeatures: {} | labels: {}\n".format(
        X_test_norm.shape, y_test_k.shape))

    m = X_train_norm.shape[0]
    w = np.random.randn(784, 1) * 0.01
    b = np.zeros((1, 10))

    # Forward Prop
    y_hat = sigmoid(np.dot(X_train_norm, w) + b)
    cost = -(1 / m) * np.sum(y_train_k * np.log(y_hat) + (1 - y_train_k) * np.log(1 - y_hat))

    # Back Prop
    dZ = y_hat - y_train_k
    dw = (1 / m) * np.sum(np.dot(X_train_norm.T, dZ))
    db = (1 / m) * np.sum(dZ)

    print(cost)
    print(dw, db)

    # # Define model hyperparameters
    # hyperparameters = {
    #     "learning_rate": 0.005,
    #     "training_iters": 2000
    # }
    #
    # # Train Logistic Regression model
    # model = SoftmaxRegression(hyperparameters)
    # model.print_model(X_train_norm, y_train_k)
    # model = model.gradient_descent(X_train_norm, y_train)



if __name__ == '__main__':
    main()
