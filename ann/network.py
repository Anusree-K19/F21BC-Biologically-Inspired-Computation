import numpy as np
from .activation import sigmoid, relu, tanh

class ANN:

    # Simple feedforward neural network with flexible layers and activations
    def __init__(self, layer_sizes, activations, seed=42):

        # make sure activations match the number of layer transitions
        assert len(activations) == len(layer_sizes) - 1, "activations mismatch"
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.weights, self.biases = [], []
        rng = np.random.default_rng(seed)                           # random generator for reproducibility
        self.init_params(rng)                                       # setup initial weights and biases

    def init_params(self, rng):

        # create small random weights and zero biases for each layer
        for i in range(len(self.layer_sizes) - 1):
            in_, out_ = self.layer_sizes[i], self.layer_sizes[i+1]
            W = rng.normal(loc=0.0, scale=0.1, size=(out_, in_))    # weights
            b = np.zeros((out_, 1))                                 # biases
            self.weights.append(W)
            self.biases.append(b)

    def act(self, z, name):

        # apply the right activation based on the given name
        if name == "relu": return relu(z)
        if name == "sigmoid": return sigmoid(z)
        if name == "tanh": return tanh(z)
        return z                                                    # linear activation (no change)

    def forward(self, x_colvec):

        # keep input as (input_dim, 1)
        assert x_colvec.ndim == 2 and x_colvec.shape[1] == 1, "x must be (in_dim, 1) column vector"

        # pass input forward through all layers to get the output
        a = x_colvec
        for W, b, act in zip(self.weights, self.biases, self.activations):
            z = W @ a + b                                           # linear combination
            a = self.act(z, act)                                    # apply activation
        return a                                                    # final output
