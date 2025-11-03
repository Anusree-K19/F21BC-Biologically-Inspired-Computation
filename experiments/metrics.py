import numpy as np

# mean absolute error – simple regression metric
def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))