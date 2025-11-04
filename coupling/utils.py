import numpy as np

# turn all weights + biases into one flat vector
def flatten_params(weights, biases):
    flat = []
    for W, b in zip(weights, biases):
        flat.append(W.ravel()); flat.append(b.ravel())
    return np.concatenate(flat)

# rebuild weights + biases from a flat vector (fixed architecture)
def unflatten_params(vec, layer_sizes):
    weights, biases = [], []
    i = 0
    for l in range(len(layer_sizes) - 1):
        in_, out_ = layer_sizes[l], layer_sizes[l+1]
        Wn = out_ * in_
        W = vec[i:i+Wn].reshape(out_, in_); i += Wn
        b = vec[i:i+out_].reshape(out_, 1); i += out_
        weights.append(W); biases.append(b)
    return weights, biases

# set network parameters from a flat vector
def set_params(net, vec):
    W, B = unflatten_params(vec, net.layer_sizes)
    net.weights, net.biases = W, B