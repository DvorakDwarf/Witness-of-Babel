#The script to train the model
#Main not a good name. Could not think of better one

import torch
import torchvision.transforms as transforms

from PIL import Image
from matplotlib import pyplot as plt

from components.data import get_loaders
from components import architecture
from components.training import training_loop

IMAGE_SIZE = 24
CHECKPOINT_NAME = f"Medium_Witness_of_Babel_{IMAGE_SIZE}"

#When training the model, I was spooked because it wouldn't overfit
#Turns out there is so much data it can't overfit, 4head accident

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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMAGE_SIZE),
    transforms.Grayscale()
    ])

train_loader, val_loader = get_loaders(img_size=IMAGE_SIZE,
    transform=transform,
    device=device)

# model = architecture.large.LargeWitness().to(device)
model = architecture.medium.MediumWitness().to(device)
# model = architecture.small.SmallWitness().to(device)

training_loop(model, train_loader, val_loader, name=CHECKPOINT_NAME)