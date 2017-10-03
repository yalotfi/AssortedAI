import sys
import os
import pprint as pp
import numpy as np

sys.path.insert(0, os.path.join(os.getcwd()))
from assort.utils import load_datasets
# from assort.regression.softmax import SoftmaxRegression


def main():
    # (X_train, y_train), (X_test, y_test) = get_iris()
    features, labels = load_datasets.get_iris()
    print(np.array(features, dtype='f8'))
    print(np.unique(np.array(labels)))
    # pp.pprint(X)
    # pp.pprint(y)

    # hyperparams = {
    #     "training_iters": 2500,
    #     "learning_rate": 0.001,
    #     "init_param_bound": 0.01
    # }
    # model = SoftmaxRegression(X.T, Y.T, hyperparams)


if __name__ == '__main__':
    main()
