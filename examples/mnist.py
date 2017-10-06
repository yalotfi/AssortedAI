import numpy as np
import time as t
import sys
import os
import pprint as pp

sys.path.insert(0, os.path.join(os.getcwd()))
from assort.utils.load_datasets import get_mnist
from assort.preprocessing.one_hot import one_hot_encode
from assort.regression.softmax import SoftmaxRegression

from assort.activations import softmax


def train():
    # Train Logistic Regression model
    cost_cache = []
    X = X_train_norm.T
    Y = y_train_k.T
    n, m = X.shape[0], X.shape[1]
    k = Y.shape[0]
    w = np.random.rand(n, k) * 0.01
    b = np.zeros((k, 1))
    for i in range(1000):
        # Forward Prop
        Z = np.dot(w.T, X) + b
        A = softmax(Z)
        cost = -np.sum(Y * np.log(A))
        # Back prop
        dZ = Y - A
        dw = -(1 / m) * np.dot(X, dZ.T)
        db = -(1 / m) * np.sum(dZ)
        # Update
        w = w - 0.0001 * dw
        b = b - 0.0001 * db
        # Save cost
        if i % 100 == 0:
            print(cost)
            cost_cache.append(cost)

    plt.plot(cost_cache)
    plt.ylabel('Training Cost')
    plt.xlabel('Training Iteration')
    plt.show()


def main():
    tic = t.time()
    (X_train, y_train), (X_test, y_test) = get_mnist(download=False,
                                                     serialize=False,
                                                     binary=False,
                                                     bin_digits=[0, 1],
                                                     flatten=True)
    toc = t.time() - tic
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
    model.show_error()
    # model.test_train(X, Y)

if __name__ == '__main__':
    main()
