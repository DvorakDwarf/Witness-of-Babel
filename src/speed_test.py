#Compare the speed of 2 models

import torch
import torchvision.transforms as transforms

import time
from tqdm import tqdm

from components import noisemaker 
from components.architecture import large
from components.architecture import medium
from components.architecture import small

#Reduce number if GPU too small
CHUNK_SIZE = 5000
IMAGE_SIZE = 24

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

model1 = medium.MediumWitness().to(device)
model2 = small.SmallWitness().to(device)

model1.load_state_dict(torch.load("data/Medium_Witness_of_Babel_24.pth"))
model2.load_state_dict(torch.load("data/Small_Witness_of_Babel_24.pth"))

models = [("Witness", model1), ("Small Witness", model2)]
noisegen = noisemaker.NoiseGen(IMAGE_SIZE)
for arch in models:
    start = time.time()
    for _ in tqdm(range(0, 1000)):
        chunk = noisegen.generate_chunk(CHUNK_SIZE).to(device)
        outputs = arch[1](chunk)

        # Simulate the stop for heartbeat at bot_search
        time.sleep(0.0001)

    elapsed = time.time() - start
    print(f"{arch[0]} finished the task in {elapsed}")
