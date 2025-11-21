import numpy as np
from ann.network import ANN
from coupling.utils import flatten_params, unflatten_params
from experiments.data_utils import load_concrete
from experiments.predict import predict_batch
from experiments.metrics import mae

# This script is only used to verify that the core modules(ANN, activation functions, parameter flattening utilities,
# and data loading) work correctly. 
# It performs sanity checks on the ANN and runs a simple random-weights baseline to confirm the pipeline runs without errors.

# Full PSO training and all experiments are handled in the test.ipynb notebook, not in this file.


# To run a series of sanity checks on the ANN implementation
def sanity_checks():
    # Small ANN with 8 inputs,  16 hidden (ReLU) and 1 output (linear)
    net = ANN([8, 16, 1], ["relu", "linear"], seed=0)

    # 1. To perform sanity check: forward on one sample
    x = np.random.rand(8, 1)                                        # To create single sample as column vector
    y = net.forward(x)
    print("1) Output shape:", y.shape)
    print("1) Output value:", y.item())

    # 2. To perform sanity check: forward on zero input
    y0 = net.forward(np.zeros((8,1)))
    print("2) zero-input output (should be approximately 0):", y0.item())

    # 3. To perform shape check: wrong input shape should raise an error
    try:
        net.forward(np.random.rand(8))                              # To create wrong shape input (not a column vector)
        print("3) shape check FAILED (should have errored)")
    except Exception as e:
        print("3) shape check ok - ", e)

    # 4. To perform configurable architecture test
    net2 = ANN([8, 8, 8, 1], ["tanh", "relu", "linear"], seed=1)
    y2 = net2.forward(np.random.rand(8,1))
    print("4) configurable architecture ok - ", y2.shape)

    # 5. To perform flatten/unflatten consistency test
    vec = flatten_params(net.weights, net.biases)
    W2, B2 = unflatten_params(vec, net.layer_sizes)

    # To replace parameters and check output stays identical
    old_out = net.forward(x)
    net.weights, net.biases = W2, B2
    new_out = net.forward(x)
    same = np.allclose(old_out, new_out)                            # To perform numeric safe comparison
    print("5) flatten/unflatten consistent: ", same)

    # 6. To perform random parameter test: different parameters should give different output
    param_count = sum(W.size + b.size for W, b in zip(net.weights, net.biases))
    print("6) total params:", param_count)

# To run a random-weights baseline on the concrete dataset
def random_weights_baseline():

    # To load data (70/30 split)
    try:
        Xtr, ytr, Xte, yte, _, _ = load_concrete("data/concrete.csv", test_ratio=0.3, seed=0)
    except FileNotFoundError:
        print("Dataset not found at data/concrete.csv")
        return

    # To build a small ANN for regression
    net = ANN([8, 16, 1], ["relu", "linear"], seed=0)
    # To perform random-weights predictions on test set
    y_pred = predict_batch(net, Xte)
    # To compute MAE
    test_mae = mae(yte, y_pred)
    print("Random-weights Test MAE:", test_mae)

# Main execution
if __name__ == "__main__":
    sanity_checks()
    random_weights_baseline()