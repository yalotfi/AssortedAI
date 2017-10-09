from assort.initializers import *
from assort.cost_functions import MeanSquaredError

_INITIALIZER_CONFIG = {
    'zeros': Zeros,
    'ones': Ones,
    'random_normal': RandomNormal
}

_COST_FUNC_CONFIG = {
    'mse': MeanSquaredError
}
