import numpy as np

# To run forward for a whole batch (row-wise X)
def predict_batch(net, X):
    predictions = []
    for i in range(X.shape[0]):
        xi = X[i].reshape(-1, 1)                    # To make column vector
        yi = net.forward(xi)
        predictions.append(yi)
    return np.vstack(predictions)                   # (n_samples, 1)