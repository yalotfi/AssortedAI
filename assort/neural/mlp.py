import numpy as np


class MultiLayerPerceptron(object):
    """docstring for MultiLayerPerceptron."""
    def __init__(self, layer_dims):
        super(MultiLayerPerceptron, self).__init__()
        self.layer_dims = layer_dims
        self.L = len(self.layer_dims)
        self.parameters = self._get_params

    def _initialize_parameters(self, layer_dims):
        parameters = {}
        for l in range(1, self.L):
            parameters['W' + str(l)] = np.random.randn(
                layer_dims[l], layer_dims[l - 1]) * 0.01
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        return parameters

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
