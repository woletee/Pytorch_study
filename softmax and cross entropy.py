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
y=np.array([1,0,0])
y_pred_good=np.array([0.7,0.2,0.1])
y_pred_bad=np.array([0.1,0.3,0.6])
l1=Cross_Entropy(y,y_pred_good)
l2=Cross_Entropy(y,y_pred_bad)
print(f'loss1 numpy:{l1:.4f}')
print(f'loss2 numpy:{l2:.4f}')

#Cross entropy using pytorch 
loss=nn.CrossEntropyLoss()
y=torch.tensor([0])
y_pred_good=torch.tensor([[2.0,1.0,0.1]])
y_pred_bad=torch.tensor([[0.5,2.0,0.3]])
l1=loss(y_pred_good,y)
l2=loss(y_pred_bad,y)
print(l1.item())
print(l2.item())
# Assuming y_pred_good and y_pred_bad are defined 1D tensors

# y_pred_good = torch.tensor([0.1, 0.2, 0.7])
# y_pred_bad = torch.tensor([0.1, 0.3, 0.6])

_, prediction1 = torch.max(y_pred_good, dim=0)
_, prediction2 = torch.max(y_pred_bad, dim=0)

print(prediction1)
print(prediction2)
