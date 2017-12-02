import numpy as np

from assort.initializers import RandomNormal
from assort.initializers import Zeros


class DeepNeuralNetwork(object):
    """docstring for DeepNeuralNetwork."""
    def __init__(self, layer_dims):
        super(DeepNeuralNetwork, self).__init__()
        self.layer_dims = layer_dims
        self.L = len(self.layer_dims)
        self.parameters = self._get_params

    def _initialize_parameters(self, layer_dims):
        """
        Initialize model parameters with random normal distribution

        Arguments
        ---------
        layer_dims : list
            Each element is the input dimension of the corresponding layer

        Returns
        -------
        dict
            Python dictionary containing initialized weights and biases for
            each layer of the model
        """
        rand_init = RandomNormal()
        zero_init = Zeros()
        parameters = {}
        for l in range(1, self.L):
            weight_shape = (layer_dims[l], layer_dims[l - 1])
            bias_shape = (layer_dims[l], 1)
            parameters['W' + str(l)] = initializer(weight_shape)
            parameters['b' + str(l)] = zero_init(bias_shape)
        return parameters

    def _linear_forward(A, W, b):
        """
        Implement a single linear transformation during forward propagation

        Arguments
        ---------
        A : ndarray
            The previous layer's activations are inputs to the current layer.
                - Size: (hh_prev, m_examples)
        W : ndarray
            The current layer's weights
                - Size: (hh_curr, hh_prev)
        b : ndarray
            The current layer's bias units
                - Size: (hh_curr, 1)

        Returns
        -------
        ndarray
            The input to an activation function of the current layer
        dict
            Python dictionary caching the previous activations, layer weights
            and bias which will be used when computing back propagation
        """
        Z = np.dot(W, A) + b
        cache = A, W, b
        return A, cache

    def _forward_activation(A_prev, W, b, activation_func):
        Z, linear_cache = self._linear_forward(A_prev, W, b)
        A = sigmoid(Z)

    @property
    def _get_params(self):
        return self._initialize_parameters(self.layer_dims)


def main():
    layers = [784, 20, 2, 20, 10]
    mlp = MultiLayerPerceptron(layers)
    for key, value in mlp.parameters.items():
        print("{} | {}".format(key, value.shape))



if __name__ == '__main__':
    main()
