import torch
import numpy as np

# Softmax function using NumPy
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

