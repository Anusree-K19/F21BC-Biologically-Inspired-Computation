import numpy as np

# The sigmoid (logistic) activation squashes values between 0 and 1
def sigmoid(x): 
    return 1/(1+np.exp(-x))

# The ReLU (Rectified Linear Unit) activation keeps positives and sets negatives to zero
def relu(x): 
    return np.maximum(0, x)

# The hyperbolic tangent (tanh) activation squashes values between -1 and 1
def tanh(x): 
    return np.tanh(x)

# The softplus activation is a smooth approximation to ReLU
def softplus(x):
    return np.log1p(np.exp(x))
