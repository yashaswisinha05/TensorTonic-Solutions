import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    x=np.asarray(x, dtype=float)
    sigma=1/(1+np.exp(-x))# Write code here
    return sigma
    pass