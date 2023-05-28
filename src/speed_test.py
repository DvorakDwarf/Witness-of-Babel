#Compare the speed of 2 models

import torch
import torchvision.transforms as transforms

from components import noisemaker 
from components.architecture import Witness
from components.small_architecture import SmallWitness

import time
from tqdm import tqdm

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

model = Witness().to(device)
small_model = SmallWitness().to(device)
model.load_state_dict(torch.load("data/Witness_of_Babel.pth"))
small_model.load_state_dict(torch.load("data/Small_Witness_of_Babel.pth"))

models = [("Witness", model), ("Small Witness", small_model)]

noisegen = noisemaker.NoiseGen()
for arch in models:
    start = time.time()
    for _ in tqdm(range(0, 1000)):
        chunk = noisegen.generate_chunk(1000).to(device)
        outputs = arch[1](chunk)

        time.sleep(0.0001)

    elapsed = time.time() - start
    print(f"{arch[0]} finished the task in {elapsed}")
