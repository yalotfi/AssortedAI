import numpy as np
import time as t

from pandas import get_dummies
from pandas.io.parsers import read_csv

from assort.regression.softmax import SoftmaxRegression


def main():
    ###################
    # Testing on Iris #
    ###################
    tic = t.time()
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    col_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    toc = t.time() - tic
    print("Download Time: {}\n".format(toc))
    iris = read_csv(url, names=col_names)
    X = np.array(iris.ix[:, :-1])
    Y = np.array(get_dummies(iris.ix[:, -1]))
    print(X.shape)
    print(Y.shape)
    hyperparams = {
        "training_iters": 2500,
        "learning_rate": 0.001,
        "init_param_bound": 0.01
    }
    model = SoftmaxRegression(X.T, Y.T, hyperparams)


if __name__ == '__main__':
    main()