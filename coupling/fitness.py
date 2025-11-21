import numpy as np
from experiments.predict import predict_batch
from experiments.metrics import mae
from ann.network import ANN
from .utils import set_params

# To create a fitness function for PSO to optimize ANN weights+biases
def make_ann_fitness(Xtr, ytr, arch, activations, seed=0):
    
    net = ANN(arch, activations, seed=seed)                         # Architecture fixed for this PSO run

    # Fitness function to set parameters, predict on train set and return MAE
    def f(vec):
        set_params(net, vec)
        y_pred = predict_batch(net, Xtr)
        return float(mae(ytr, y_pred))
    return f