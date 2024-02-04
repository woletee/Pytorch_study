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

