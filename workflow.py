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
