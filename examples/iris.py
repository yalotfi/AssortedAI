import sys
import os
import numpy as np


sys.path.insert(0, os.path.join(os.getcwd()))
from assort.utils import load_datasets
from assort.preprocessing import feature_scaling as norm
from assort.preprocessing import one_hot as ova
from assort.linear.logistic import SoftmaxRegression


def main():
    # Load Iris Dataset
    (X_train, y_train), (X_test, y_test) = load_datasets.get_iris(99)

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    # Normalization
    X_train_norm = norm.standardize(X_train)
    X_test_norm = norm.standardize(X_test)

    # One-Hot Encoding
    iris_classes = int(np.max(y_train))
    y_train_ova = ova.one_hot_encode(y_train, iris_classes)
    y_test_ova = ova.one_hot_encode(y_test, iris_classes)

    # Train and evaluate the model
    model = SoftmaxRegression(epochs=5000, lr=0.00005, lmda=0.001)
    model.fit(X_train_norm, y_train_ova)
    print(model.evaluate(X_train_norm, y_train))
    print(model.evaluate(X_test_norm, y_test))


if __name__ == '__main__':
    main()
