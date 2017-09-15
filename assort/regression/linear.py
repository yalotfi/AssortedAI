import numpy as np

# Implement Linear Regression trained with SGD
# h(x) = np.dot(w.T, X) + b
# J(w, b) = (1 / 2) * np.sum(np.square(h(w, b) - y))

def hypothesis(w, b, X):
    return np.dot(w, X) + b

def compute_grads(y_hat, y, X):
    pass
