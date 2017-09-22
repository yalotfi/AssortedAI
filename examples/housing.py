import sys
import os

# Add AssortedAI to system path for development
sys.path.insert(0, os.path.join(os.getcwd()))
from assort.utils import preprocessing
from assort.regression.linear import LinearRegression
from assort.utils.load_datasets import get_housing


def main():
    X_train, y_train, X_test = get_housing()
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)


if __name__ == '__main__':
    main()
