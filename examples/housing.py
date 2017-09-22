import sys
import os
import numpy as np

# Add AssortedAI to system path for development
sys.path.insert(0, os.path.join(os.getcwd()))
from assort.regression.linear import compute_cost


def main():
    # Load data
    fname = 'housing_uni.txt'
    fpath = os.path.join('examples', 'datasets', 'housing', fname)
    data = np.loadtxt(fpath, delimiter=',')
    # Prepare data
    X_train = data[:, 0]
    y_train = data[:, 1].reshape((data.shape[0], 1))
    X_train = np.c_[np.ones(data.shape[0]), data[:, 0]]
    theta = np.array([[0], [0]])
    print(X_train.shape, y_train.shape, theta.shape)
    cost1, cost2 = compute_cost(X_train, y_train, theta)
    print(cost1, cost2)


if __name__ == '__main__':
    main()
