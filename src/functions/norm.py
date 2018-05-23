import numpy as np

def L2(W):
    w = W.reshape((W.shape[0], -1)).data
    return np.array([max(abs(np.linalg.svd(w)[1]))])
