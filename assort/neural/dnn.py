import numpy as np

from assort.initializers import RandomNormal, Zeros
from assort.activations import relu, sigmoid


class DeepNeuralNetwork(object):
    """docstring for DeepNeuralNetwork."""
    def __init__(self, layer_dims):
        super(DeepNeuralNetwork, self).__init__()
        self.layer_dims = layer_dims
        self.L = len(self.layer_dims)
        self.parameters = self._get_parameters()

    @property
    def _get_parameters(self):
        return self._initialize_parameters(self.layer_dims)

    def _initialize_parameters(self, layer_dims):
        """Initialize model parameters with random normal distribution

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
            parameters['W' + str(l)] = rand_init(weight_shape)
            parameters['b' + str(l)] = zero_init(bias_shape)
        return parameters

    def _linear_forward(A_prev, W, b):
        """Implement a single linear transformation during forward propagation

        Arguments
        ---------
        A_prev : ndarray
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
        tuple
            Cache the previous activations and layer parameters which will be
            used to compute back propagation
        """
        W, b = parameters['W'], parameters['b']
        Z = np.dot(W, A_prev) + b
        cache = (A_prev, W, b)
        return Z, cache

    def _forward_activation(A_prev, W, b, activation):
        """Implement a single forward activation layer in a neural network.

        Arguments
        ---------
        A_prev : ndarray
            The previous layer's activations are inputs to the current layer.
                - Size: (hh_prev, m_examples)
        W : ndarray
            The current layer's weights
                - Size: (hh_curr, hh_prev)
        b : ndarray
            The current layer's bias units
                - Size: (hh_curr, 1)
        activation : str
            String defining which activation function to use

        Returns
        -------
        ndarray
            Input to the next layer's A_prev
        tuple
            Cache containing the linear cache plus the linear input to this
            layer's activation
        """
        Z, linear_cache = self._linear_forward(A_prev, W, b)
        if activation == 'sigmoid':
            A = sigmoid(Z)
        elif activation == 'relu':
            A = relu(Z)
        cache = (linear_cache, Z)
        return A, cache

    def _feed_forward(self, X, parameters):
        """Implement a full forward pass of the neural network

        Argument
        --------
        X : ndarray
            Input data with shape (n_features, m_examples)

        Returns
        -------
        ndarray
            Final layer activation output
        list
            List of caches from each layer
        """
        # Define helper variables
        L = self.L  # Number of layers in the DNN
        A = X  # Input data, X, is just the 'first' layer activation

        # Implement [LINEAR -> RELU] for (L - 1) layers, storing each cache
        caches = []
        for l in range(1, L):
            A_prev = A
            W_l = parameters['W' + str(l)]
            b_l = parameters['W' + str(l)]
            A, cache = self._forward_activation(A_prev, W_l, b_l, 'relu')
            caches.append(cache)

        # Implement [LINEAR -> SOFTMAX] for final layer, L, and add final cache
        W_L = parameters['W' + str(L)]
        b_L = parameters['W' + str(L)]
        AL, cache = self._forward_activation(A, W_L, b_L, 'softmax')
        caches.append(cache)
        return AL, caches

    def _linear_backward(dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[0]
        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db


def main():
    layers = [784, 20, 2, 20, 10]
    nn = DeepNeuralNetwork(layers)
    for key, value in nn.parameters.items():
        print("{} | {}".format(key, value.shape))


if __name__ == '__main__':
    main()
