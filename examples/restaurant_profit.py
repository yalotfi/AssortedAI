import sys
import os
import numpy as np

# Add AssortedAI to system path for development
sys.path.insert(0, os.path.join(os.getcwd()))
from assort.regression.linear import compute_cost


def main():
    # Load data
    fname = 'chain_profits.txt'
    fpath = os.path.join('examples', 'datasets', fname)
    data = np.loadtxt(fpath, delimiter=',')
    # Prepare data
    m, n = data.shape[0], 1
    X = data[:, 0].reshape((m, n))
    y = data[:, 1].reshape((m, 1))
    theta = np.array([[0], [0]])
    print(X.shape, y.shape, theta.shape)
    cost, grad = compute_cost(X, y, theta)
    print("Cost: {}".format(cost))
    print("Grad: \n{}".format(grad))
    print(grad.shape)


if __name__ == '__main__':
    main()
