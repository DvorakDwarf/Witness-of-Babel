import torch
import torchvision.transforms as transforms

from PIL import Image
from matplotlib import pyplot as plt

from components.data import get_loaders
from components.architecture import Witness
from components.training import training_loop

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
    transforms.Grayscale()
    ])

train_loader, val_loader = get_loaders("/home/titleless/m2/datasets/tiny-imagenet/",
transform=transform,
device=device)

# for x, y in train_loader:
#     # x = torch.round(x*255)
    
#     print(x[0][0])
#     print(x.shape)

#     x = x.cpu()
#     plt.imshow(x[0].reshape(64, 64, 1))
#     plt.show()

model = Witness().to(device)

training_loop(model, train_loader, val_loader)