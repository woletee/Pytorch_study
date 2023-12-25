import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
class WineDataset(Dataset):
    def __init__(self):
        # Data loading
        xy = np.loadtxt('./wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]
    def __getitem__(self, index):
        # Corrected return statement
        return self.x[index], self.y[index]
    def __len__(self):
        return self.n_samples
# Create dataset instance
dataset = WineDataset()
# DataLoader setup
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)
# Training loop
num_epochs = 2
total_samples = len(dataset)


