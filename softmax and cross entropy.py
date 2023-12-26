import torch
import numpy as np

# Softmax function using NumPy
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# Input array
x_np = np.array([2.0, 1.0, 0.1])
# Applying the softmax function with NumPy
outputs_np = softmax(x_np)
print('Softmax using NumPy:', outputs_np)
# Softmax using PyTorch
x_torch = torch.tensor([2.0, 1.0, 0.1])
outputs_torch = torch.softmax(x_torch, dim=0)
print('Softmax using PyTorch:', outputs_torch)
#Calculating the cross entropy using the numpy 
#Cross entropy using the numpy 
def Cross_Entropy(actual,predicted):
    loss=-np.sum(actual*np.log(predicted))
    return loss
