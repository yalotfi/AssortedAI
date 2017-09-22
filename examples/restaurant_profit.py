import sys
import os

# Add AssortedAI to system path for development
sys.path.insert(0, os.path.join(os.getcwd()))
from assort.regression.linear import LinearRegression
from assort.utils.load_datasets import get_profit


def main():
    # Load Data
    (X_train, y_train, X_test) = get_profit()

    # Define model hyperparameters
    hyperparameters = {
        "learning_rate": 0.01,
        "epochs": 1500
    }

    # Fit Linear Regression model
    model = LinearRegression(hyperparameters)
    model = model.train_sgd(X_train, y_train)
    pred = model.predict(X_test)
    print(pred)

    thetas = model.trained_params["theta"]
    print("\nParameters for Univariate OLS:\n{}\n".format(thetas))


if __name__ == '__main__':
    main()
