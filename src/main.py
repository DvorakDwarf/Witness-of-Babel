import torch
import torchvision.transforms as transforms

from PIL import Image
from matplotlib import pyplot as plt

import data 
from architecture import

#Check what device to use
use_cuda = torch.cuda.is_available()
use_mps = torch.backends.mps.is_available()

device = "cpu"
if use_cuda == True:
    device = "cuda"
elif use_mps == True:
    device = "mps"

device = torch.device(device)
print(f"Device is {device}")

torch.set_default_device(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale()
    ])

train_loader, val_loader = data.get_loaders("/home/titleless/m2/datasets/tiny-imagenet/", transform=transform)
