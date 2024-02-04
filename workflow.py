#what we are going to do
#1 prepare and load dat
#2 build a model
#3 fitting the model to the data(training)
#4 making predictions and making inference
#5 saving and loading the model
#6 putting it all together

import torch
from torch import nn  ##nn contans all of the building blocks for the neural networks

import matplotlib.pyplot as plt

#first step is to load and prepare the data
#the data can be anything in machine learning like images, video, numbers, podacasts, protein structures, text and so on.
#Machine learning is basically about two things 
#the first is to represent your data on your hand whatever its type is into number representation
#the next step is to build a model or reload build model to learn the best reperesentation of the underlying data
weight=0.7
bias=0.3


start=0
end=1
step=0.02
X=torch.arange(start, end, step).unsqueeze(dim=1)
y=weight*X + bias

X[:10], y[:10]

#now we have got the data for training our model
#but we need to split the data into training and test 
train_split=int(0.8* len(X)) #80 % of the data is used for the trainng 
X_train , y_train=X[:train_split], y[:train_split]
X_test, y_test=X[:train_split], y[:train_split]
len(X_train), len(y_train), len(X_test),len(y_test)

#lets function to visualize the data that we have generated 

def plot_predictions(train_data=X_train, train_labels=y_train,test_data=X_test, test_labels=y_test,predictions=None):
       plt.figure(figsize=(10,7))
       
       #plot training data 
       plt.scatter(train_data, train_labels, c="b", s=4, label="Trainng data")
       
       if predictions is not None:
           plt.scatter(test_data, predictions, c="r",s=4, label="predictions")
       
       plt.legend(prop={"size": 14})
plot_predictions()

#2 Next we will build the model
#nn.module is the super class for all neural netwroks 
#If you are implementing a neural network it shoul be a subclass of nn.module
#all nn.module class requires forward method to be Implemneted 
#torch.optim contains the methods on how to improve the parametrs with in the nn.module to better reperesent the input data

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights=nn.Parameter(torch.randn(1, dtype=torch.float, requires_grad=True))
        self.bias=nn.Parameter(torch.randn(1, dtype=torch.float, requires_grad=True))
    def forward(self, x: torch.Tensor):
        return self.weights *x +self.bias
#Now lets create a model instance using the above class that we have created before 
#we can check the parameters of the class instance that we have created using .parametrs() method
#checking the contents of a pytorch model
torch.manual_seed(42) #b/c .parametrs are randomly initalized
model_0=LinearRegressionModel() #create instacne of the model class 
print(list(model_0.parameters()))  #check the nn.parametrs within the nn.module class that we have created before 
print(model_0.state_dict())

#next we will make predictions using torch.inference_mode()
#make predictions with this model
with torch.inference_mode():
    y_preds=model_0(X_test)

#check the predictions
print("the number of testing smapples:{len(X_tes)}")
print("the number of predictions made :{{len(y_preds)}")
print(f"predicted values:\n{y_preds}")

#3 train the model
loss_fn=nn.L1Loss()
optim=torch.optim.SGD(params=model_0.parameters(), lr=0.01)

#Now lets create the traing loop for training our model
epochs=100
train_loss_values=[]
test_loss_values=[]
epoch_count=[]

for epoch in range (epochs):
    model_0.train() #put the model in training mode
    y_pred=model_0(X_train) #forward pass on the train data using the method defined before
    loss=loss_fn(y_pred,y_train)
    optim.zero_grad() # zero grad of the optimizer
    loss.backward()
    optim.step()  #progrss the optimizer 
    
    model_0.eval() #testing the model
    
    with torch.inference_mode():
        test_pred=model_0(X_test)
        test_loss=loss_fn(test_pred,y_test.type(torch.float))
        
        if epoch % 10==0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy)
            print(f"Epoch: {epoch}| MAE Train Loss: {loss}| MAE Loss:{test_loss}")
#plottng the curves  
plt.plot(epoch_count, train_loss_values, label="Train Loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and loss curves")
plt.ylabel("loss")
plt.xlabel("Epochs")
plt.legend()
