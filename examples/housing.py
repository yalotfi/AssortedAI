import sys
import os

# Add AssortedAI to system path for development
sys.path.insert(0, os.path.join(os.getcwd()))
from assort.utils import norm
from assort.regression.linear import LinearRegression
from assort.utils.load_datasets import get_housing


def main():
    X_train, y_train, X_test = get_housing()
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)

    # Perform feature scaling on X_train and X_test
    X_norm = norm.standardize(X_train)
    X_test = norm.standardize(X_test)

    # Define model hyperparamters for LinReg
    hyperparameters = {
        "learning_rate": 0.03,
        "epochs": 100
    }

    # Fit Linear Regression model
    model = LinearRegression(hyperparameters)
    model = model.train_sgd(X_norm, y_train, print_cost_freq=10)

    # Visualize training error over each iteration
    model.plot_error()

    # Make predictions with trained model
    pred = model.predict(X_test)
    print(pred)


if __name__ == '__main__':
    main()
