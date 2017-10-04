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
    # X.shape: (60000, 784)
    w = np.random.randn(10, 784) * 0.01
    b = np.zeros((10, 1))

    # Forward Prop
    y_hat = sigmoid(np.dot(w, X_train_norm.T) + b)
    y = y_train_k.T
    print(y_hat.shape)
    print(y.shape)
    cost = -(1 / m) * np.sum(y * np.log(y_hat), axis=0)
    # cost = -(1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    # print(cost)

    # # Back Prop
    # dZ = y_hat - y
    # print(dZ.shape)
    # dw = (1 / m) * np.sum(np.dot(dZ,X_train_norm))
    # db = (1 / m) * np.sum(dZ)
    #
    # print(dw, db)

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
