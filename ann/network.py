import numpy as np
from .activation import sigmoid, relu, tanh, softplus

class ANN:

    # Simple feedforward neural network with flexible layers and activations
    def __init__(self, layer_sizes, activations, seed=42):

        # To make sure activations match the number of layer transitions
        assert len(activations) == len(layer_sizes) - 1, "activations mismatch"
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.weights, self.biases = [], []
        rng = np.random.default_rng(seed)                           # Random generator for reproducibility
        self.init_params(rng)                                       # Setup initial weights and biases

    def init_params(self, rng):

        # To create small random weights and zero biases for each layer
        for i in range(len(self.layer_sizes) - 1):
            in_, out_ = self.layer_sizes[i], self.layer_sizes[i+1]
            W = rng.normal(loc=0.0, scale=0.1, size=(out_, in_))    # weights
            b = np.zeros((out_, 1))                                 # biases
            self.weights.append(W)
            self.biases.append(b)

    def act(self, z, name):

        # To apply the right activation based on the given name
        if name == "relu": return relu(z)
        if name == "sigmoid": return sigmoid(z)
        if name == "tanh": return tanh(z)
        if name == "softplus":  return softplus(z)
        return z                                                    # Linear activation

    def forward(self, x_colvec):

        # To keep input as (input_dim, 1)
        assert x_colvec.ndim == 2 and x_colvec.shape[1] == 1, "x must be (in_dim, 1) column vector"

        # To pass input forward through all layers to get the output
        a = x_colvec
        for W, b, act in zip(self.weights, self.biases, self.activations):
            z = W @ a + b                                           # Linear combination
            a = self.act(z, act)                                    # Apply activation
        return a                                                    # Final output
