from assort.initializers import *
from assort.cost_functions import *

_INITIALIZER_CONFIG = {
    'zeros': Zeros,
    'ones': Ones,
    'random_normal': RandomNormal
}

_COST_FUNC_CONFIG = {
    'mean_squared_error': MeanSquaredError,
    'binary_cross_entropy': BinaryCrossEntropy,
    'categorical_cross_entropy': CategoricalCrossEntropy
}
