import numpy as np
import pandas as pd

# Loads CSV, does a 70/30 split, standardises X using train stats
def load_concrete(path="data/concrete.csv", test_ratio=0.3, seed=42, scale_y=False):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].to_numpy(dtype=float)                                       # 8 features
    y = df.iloc[:, -1].to_numpy(dtype=float).reshape(-1, 1)                         # target

    # Shuffle and split
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    cut = int(len(X) * (1 - test_ratio))
    tr_idx, te_idx = idx[:cut], idx[cut:]

    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]

    # Standardise X with train mean/std (avoid data leakage)
    x_mean = Xtr.mean(axis=0, keepdims=True)
    x_std  = Xtr.std(axis=0, keepdims=True) + 1e-8
    Xtr = (Xtr - x_mean) / x_std
    Xte = (Xte - x_mean) / x_std

    # Scale y if needed, keep raw for now
    if scale_y:
        y_mean = ytr.mean(axis=0, keepdims=True)
        y_std  = ytr.std(axis=0, keepdims=True) + 1e-8
        ytr = (ytr - y_mean) / y_std
        yte = (yte - y_mean) / y_std
    else:
        y_mean = None; y_std = None

    return (Xtr, ytr, Xte, yte, (x_mean, x_std), (y_mean, y_std))
