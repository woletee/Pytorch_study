import torch
print(torch.__version__)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.is_available())
#Tsensor basics 
x=torch.empty(3)
print(x)
x=torch.rand(2,2)
print(x)
x=torch.rand(2,2)
y=torch.rand(2,2)
z=x+y
print(z)
