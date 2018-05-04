import sys
import os

# Add AssortedAI to system path for development
sys.path.insert(0, os.path.join(os.getcwd()))
from assort.utils.load_datasets import get_housing
from assort.preprocessing import standardize
from assort.optimizers import GradientDescent
from assort.regression.linear import LinearRegression

def main():
    X_train, y_train, X_test = get_housing()
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)

    # Perform feature scaling on X_train and X_test
    X_norm = norm.standardize(X_train)
    X_test = norm.standardize(X_test)

    # Fit Linear Regression model
    model = LinearRegression(X_norm, y_train)
    sgd = GradientDescent(learning_rate=0.03, epochs=100)
    model = model.fit(sgd, print_cost_freq=10)

    # Visualize training error over each iteration
    model.plot_error()

    # Make predictions with trained model
    pred = model.predict(X_test)
    print(pred)


if __name__ == '__main__':
    main()
