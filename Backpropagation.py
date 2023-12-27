import torch
import torch.nn as nn
import torch.optim as optim

# 1. Data preparation
# Simple dataset (X: input, Y: target)
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
Y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# 2. Model definition
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(1, 5)  # One input feature, 5 hidden units
        self.fc2 = nn.Linear(5, 1)  # 5 hidden units, one output

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
model = SimpleNet()

# 3. Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
