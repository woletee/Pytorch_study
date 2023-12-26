import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784  # 28x28
hidden_size = 100
num_classes = 10  # MNIST has 10 classes (digits 0-9)
num_epochs = 2
batch_size = 100
learning_rate = 0.001
