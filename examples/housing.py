import sys
import os
import numpy as np

# Add AssortedAI to system path for development
sys.path.insert(0, os.path.join(os.getcwd()))
from assort.regression.linear import compute_cost


def main():
    # Load data
    fname = 'house_prices.txt'
    fpath = os.path.join('examples', 'datasets', fname)
    data = np.loadtxt(fpath, delimiter=',')
    # Prepare data
    # X = np.c_[np.ones(data.shape[0]), data[:, 0]]
    X = data[:, 0:2]
    y = data[:, -1].reshape((data.shape[0], 1))
    theta = np.array([[0], [0], [0]])
    print(X.shape, y.shape, theta.shape)
    cost, grad = compute_cost(X, y, theta)
    print("Cost: {}".format(cost))
    print("Grad: \n{}".format(grad))
    print(grad.shape)


if __name__ == '__main__':
    main()
