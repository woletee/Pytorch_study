#This Autograd package is important in pytorch that will help us to calculate gradiant in pytorch 
#Gradiants are essentils for our models optemization 
#pytorch provides us autograd for calculating gradiant descent so we just have to know how to use it.
import torch 
x=torch.rand(3)  # 3 represnts different 3 random values 
print(x)
#Now lets say we want to calculate the gradiant of some function with respect to the varaibel x
#what we have to do is we must specifiy the arugment required to be true  by default it is false 
x=torch.rand(3, requires_grad=True)
#whenever we calculate a computation with this tensor x pytorch will create for us a computational graph
y=x+2 #this will create the compuational graph 
z=y*y*2
z=z.mean()
print(z)
z.backward()
print(x.grad)
#but if we didinot make the grad function true then calling backward will result in an error.
#in the background it will create jacobian matrix with the partial dereivatives and then we will multiply this with
#a grdain vector to get the result using the chain rule.
y=torch.tensor([0.1,1.0,0.001], dtype=torch.float32)
z.backward()
print(x.grad)
