import torch
from torch.utils.data import Dataset, DataLoader

import PIL
import os
import numpy as np

#[1., 0.] = Real image
#[0., 1.] = Noise

#We create an equal amount of images for noise one the spot
#Bit of a clunky solution
class RealSet(Dataset):
    def __init__(self, root_dir, transform=None):

        self.all_paths = os.listdir(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.all_paths)*2

    def __getitem__(self, idx):
        #Called on the part where we should have noise
        if idx >= len(self.all_paths):
            image = torch.randn(64, 64, 3)*255 #This is noise
            label = torch.Tensor([0., 1.])
        else:
            image = PIL.imread(self.all_paths[idx])
            label = torch.Tensor([1., 0.])

            if self.transform:
                image = self.transform(image)

        return image, label