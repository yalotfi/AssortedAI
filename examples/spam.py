import sys
import os
import numpy as np


sys.path.insert(0, os.path.join(os.getcwd()))
from assort.utils import load_datasets
from assort.preprocessing.norm import standardize
from assort.regression.logistic import LogisticRegression


def main():
    # features, labels = load_datasets.get_spam(99)
    # print(np.max(labels[:-1300]))

    # Load Spam Dataset
    (X_train, y_train), (X_test, y_test) = load_datasets.get_spam(99)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    print(np.max(y_test))
    # # Normalization
    # X_train_norm = standardize(X_train)
    # X_test_norm = standardize(X_test)
    #
    # # Train and evaluate the model
    # model = LogisticRegression(epochs=5000, lr=0.005, lmda=0.01)
    # model.fit(X_train_norm, y_train)
    # print(model.evaluate(X_train_norm, y_train))
    # print(model.evaluate(X_test_norm, y_test))
    # y_pred = model.predict(X_test_norm)
    # print(y_pred[:100])
    # print(np.argmax(y_test[:1000], axis=1))


if __name__ == '__main__':
    main()
