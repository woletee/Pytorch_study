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
