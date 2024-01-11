import torch
import torch.optim as optim
import matplotlib.pyplot as plt
#we have to create dummy input of x and then dummy output of y
x=torch.randn(100,5)
y= torch.randint(0, 2, (100,1)).type(torch.FloatTensor) 
#next we have to define the requirements for the model and then we will store it in a variable named model
#basically the above coe we have writtten is for  single layer but if 
#we used two activation functions then it would be for muliptle layers
