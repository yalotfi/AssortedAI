import sys
import os
import numpy as np


sys.path.insert(0, os.path.join(os.getcwd()))
from assort.utils import load_datasets
from assort.preprocessing import feature_scaling as norm
from assort.linear.logistic import LogisticRegression


def main():
    # Load Spam Dataset
    (X_train, y_train), (X_test, y_test) = load_datasets.get_spam(99)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    # Normalization
    X_train_norm = norm.standardize(X_train)
    X_test_norm = norm.standardize(X_test)

    # Train and evaluate the model
    model = LogisticRegression(epochs=5000, lr=10e-5, lmda=10e-5)
    model.fit(X_train_norm, y_train)
    print(model.evaluate(X_train_norm, y_train))
    print(model.evaluate(X_test_norm, y_test))


if __name__ == '__main__':
    main()
