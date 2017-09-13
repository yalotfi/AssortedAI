import numpy as np
import time as t

from pandas.io.parsers import read_csv

from assort.utils.preprocessing import one_hot_encode
from assort.utils.preprocessing import normalize


def preprocess(train_path, test_path):
    # Read in DataFrames
    train_csv = read_csv(train_path)
    test_csv = read_csv(test_path)

    # Helper vars
    m = train_csv.shape[0] - 1  # should be 784 from the 28x28 resolution
    n = np.max(train_csv.ix[:, 0]) + 1  # should be 10 from number of classes

    # Pull train/test feature matrices from DFs
    X_train = train_csv.ix[:, 1:].values.astype('float32')
    X_test = test_csv.values.astype('float32')  # Kaggle excludes test labels

    # Normalize features to prevent over/under flow
    X_train_norm = normalize(X_train)
    X_test_norm = normalize(X_test)

    # Encode train/test labels into matrices of shape: (m, n)
    y_train = one_hot_encode(train_csv.ix[:, 0].values.astype('int32'), n)

    # Return training tuple for (feature, label) set and test features
    return (X_train_norm, y_train), X_test_norm

def main():
    ######################
    ## Testing on MNIST ##
    ######################
    train = './data/train.csv'
    test = './data/test.csv'
    tic = t.time()
    (X_train, Y_train), X_test = preprocess(train, test)
    toc = t.time() - tic
    print("Preprocessing Time: {}\n".format(toc))
    print("\tTrain Set:\nfeatures: {} | labels: {}".format(X_train.shape, Y_train.shape))
    print("\tTest Set:\nfeatures: {}\n".format(X_test.shape))


if __name__ == '__main__':
    main()
