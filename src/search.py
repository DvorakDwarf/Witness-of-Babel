#Basically the same at bot_search but no discord for notifications
#I now realize it's almost the same script. Could be put into one


import torch

import requests
import os
from matplotlib import pyplot as plt
import time
import asyncio

from components import noisemaker
from components.architecture import large
from components.architecture import medium
from components.architecture import small
from components import logger

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

witness = medium.MediumWitness().to(device)
witness.load_state_dict(torch.load("data/Medium_Witness_of_Babel_24.pth", map_location=device))
noisegen = noisemaker.NoiseGen(IMAGE_SIZE)
HQ = logger.Logger(bot=False)

async def search():
    while True:
        print("ready")

        while True:
            chunk = noisegen.generate_chunk(CHUNK_SIZE).to(device)
            outputs = witness(chunk)

            # #Uncomment to visualize data
            # plt.imshow(chunk[0].cpu().reshape(32, 32, 1))
            # plt.show()
            
            await HQ.log_anomalies(chunk, outputs)

asyncio.run(search())