import sys
import os
import numpy as np


sys.path.insert(0, os.path.join(os.getcwd()))
from assort.utils import load_datasets
from assort.preprocessing.norm import standardize
from assort.regression.logistic import LogisticRegression


def main():
    # Load Spam Dataset
    (X_train, y_train), (X_test, y_test) = load_datasets.get_spam(99)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    X_train_norm = standardize(X_train)
    X_test_norm = standardize(X_test)

    model = LogisticRegression(epochs=1000, lr=0.005, lmda=0.01)
    model.fit(X_train_norm, y_train)
    print(model.evaluate(X_train_norm, y_train))
    print(model.evaluate(X_test_norm, y_test))


if __name__ == '__main__':
    main()
