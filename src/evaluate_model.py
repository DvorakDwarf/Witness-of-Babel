import torch
import torchvision.transforms as transforms

from components import data 
from components.architecture import Witness
from components.small_architecture import SmallWitness
from components import training

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

_, val_loader = data.get_loaders("/home/titleless/m2/datasets/tiny-imagenet/", 
transform=transform,
device=device)

# model = Witness().to(device)
model = SmallWitness().to(device)
model.load_state_dict(torch.load("data/Small_Witness_of_Babel.pth"))

training.validate_accuracy(model, val_loader)
