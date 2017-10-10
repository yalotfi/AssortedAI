import numpy as np


class CostFunction(object):
    """CostFunction: Base class for all cost functions

    Performs basic error handling and inits common attributes
    """

    def __init__(self, y_train, y_hat, X_train):
        super(CostFunction, self).__init__()
        self._check_array_dims(y_train, y_hat, X_train)
        self.y_train = y_train
        self.y_hat = y_hat
        self.loss = self.y_hat - self.y_train
        self.X_train = X_train
        self.m = X_train.shape[0]

    def _check_array_dims(self, y_train, y_hat, X_train):
        try:
            assert(y_train.shape == y_hat.shape)
            assert(y_train.shape[0] == X_train.shape[0])
        except ValueError:
            print("Array dimensions must match!\n\
                   y_train: (m, 1),\n\
                   y_hat: (m, 1),\n\
                   X_train: (m, n)\n")


class MeanSquaredError(CostFunction):
    """MeanSquaredError

    Arguments:
        y_train -- actual labels -  vector with shape (m, 1)
        y_hat -- predicted labels - vector with shape (m, 1)
        X -- feature matrix, X, with shape (m, n)

    Properties:
        get_cost: Returns error between predicted and actual labels
        get_grad: Returns gradient of cost with respect to parameters

    Raises:
        ValueError: Check dimensions of input arrays
    """

    def __init__(self, y_train, y_hat, X_train):
        super().__init__(y_train, y_hat, X_train)

    @property
    def get_cost(self):
        return (1 / (2 * self.m)) * np.sum(np.square(self.loss))

    @property
    def get_grad(self):
        return (1 / self.m) * np.dot(self.X_train.T, self.loss)


class BinaryCrossEntropy(CostFunction):
    """BinaryCrossEntropy

    Arguments:
        y_train -- actual labels -  vector with shape (m, 1)
        y_hat -- predicted labels - vector with shape (m, 1)
        X -- feature matrix, X, with shape (m, n)

    Properties:
        get_cost: Returns error between predicted and actual labels
        get_grad: Returns gradient of cost with respect to parameters

    Raises:
        ValueError: Check dimensions of input arrays
    """

    def __init__(self, y_train, y_hat, X_train):
        super().__init__(y_train, y_hat, X_train)

    @property
    def get_cost(self):
        case_true = (1 - self.y_train) * np.log(1 - self.y_hat)
        case_false = self.y_train * np.log(self.y_hat)
        return -(1 / self.m) * np.sum(case_true + case_false)

    @property
    def get_grads(self):
        dw = (1 / self.m) * np.dot(self.X_train.T, self.loss)
        db = (1 / self.m) * np.sum(self.loss)
        grads = {
            "dw": dw,
            "db": db
        }
        return grads


class CategoricalCrossEntropy(CostFunction):
    """CategoricalCrossEntropy

    Arguments:
        y_train -- actual labels -  vector with shape (m, 1)
        y_hat -- predicted labels - vector with shape (m, 1)
        X -- feature matrix, X, with shape (m, n)

    Properties:
        get_cost: Returns error between predicted and actual labels
        get_grad: Returns gradient of cost with respect to parameters

    Raises:
        ValueError: Check dimensions of input arrays
    """

    def __init__(self, y_train, y_hat, X_train):
        super().__init__(y_train, y_hat, X_train)

    @property
    def get_cost(self):
        return -(1 / self.m) * np.sum(self.y_hat * np.log(self.y_label))

    @property
    def get_grads(self):
        dZ = self.y_label - self.y_hat
        dw = -(1 / self.m) * np.dot(self.X_train.T, dZ)
        db = -(1 / self.m) * np.sum(dZ)
        grads = {
            "dw": dw,
            "db": db
        }
        return grads
