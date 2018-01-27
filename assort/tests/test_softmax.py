import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.getcwd()))
from assort.activations import softmax

# Testing softmax function - correctly produces probability distribution
print(softmax([[1.2, 0.9, 0.4],
               [-3.4, 0, -10]]))
print(np.sum(softmax([[1.2, 0.9, 0.4],
                      [-3.4, 0., -10]]), axis=0))
