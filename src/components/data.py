#Code for the dataset as well as a function to create data loaders

import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import os
import numpy as np

BATCH_SIZE = 64
DATA_WORKERS = 0
root_dir = "/home/titleless/m2/datasets/tiny-imagenet/"

#[1., 0.] = Real image
#[0., 1.] = Noise

#We create an equal amount of images for noise one the spot
#Bit of a clunky solution
class RealSet(Dataset):
    def __init__(self, device, img_size, transform):

        self.all_paths = os.listdir(root_dir)
        self.root_dir = root_dir
        self.device = device
        self.transform = transform

        self.noise = torch.rand((len(self.all_paths), 1, img_size, img_size))

    #Same number of noise as there are real images
    def __len__(self):
        return len(self.all_paths)*2

    def __getitem__(self, idx):
        #Called on the part where we should have noise
        if idx >= len(self.all_paths):
            image = self.noise[idx - len(self.all_paths)]
            label = torch.Tensor([0.05, 0.95]).to(self.device)
        else:
            image = Image.open(self.root_dir + "/" + self.all_paths[idx])
            image = self.transform(image).to(self.device)
            label = torch.Tensor([0.95, 0.05]).to(self.device)

        return image.to(self.device), label

def get_loaders(device, img_size, transform): 
    train_dataset = RealSet(root_dir + "/train", device, img_size, transform)
    val_dataset = RealSet(root_dir + "/val", device, img_size, transform)

    train_loader = DataLoader(train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=DATA_WORKERS,
        shuffle=True
        )
    val_loader = DataLoader(val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=DATA_WORKERS
        )

    return train_loader, val_loader