import torch 
import torch.nn as nn
import torch.nn.functional as F
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = nn.ReLU()(out)
        out = self.linear2(out)
        out = nn.Sigmoid()(out)
        return out
