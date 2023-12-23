#This Autograd package is important in pytorch that will help us to calculate gradiant in pytorch 
#Gradiants are essentils for our models optemization 
#pytorch provides us autograd for calculating gradiant descent so we just have to know how to use it.
import torch 
x=torch.rand(3)
print(x)
#Now lets say we want to calculate the gradiant of some function with respect to the varaibel x
#what we have to do is we must specifiy the arugment required to be true 
x=torch.rand(3, requires_grad=True)
#whenever we calculate a computation with this tensor x pytorch will create for us a computational graph
y=x+2 #this will create the compuational graph 
 
