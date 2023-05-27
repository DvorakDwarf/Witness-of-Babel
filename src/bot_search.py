import torch

import discord
from discord.ext import commands
import requests
import os
from dotenv import load_dotenv
from matplotlib import pyplot as plt
import time

import noisemaker
from architecture import Witness


while True:
    if count >= 100000:
        print(time.time() - start)
        break

    chunk = noisegen.generate_chunk(1000).to(device)
    outputs = witness(chunk)
    
    for idx, prediction in enumerate(outputs):
        if prediction[0] > 0.3:
            example = chunk[0].cpu()
            plt.imshow(example.reshape(64, 64, 1))
            plt.show()
            
            success = chunk[idx].cpu()
            plt.imshow(success.reshape(64, 64, 1))
            plt.show()
    
    count += 1000
    # _, truth = torch.max(labels, dim=1)



intents = discord.Intents.all()
bot = commands.Bot(command_prefix='$', intents=intents)

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

witness = Witness().to(device)
witness.load_state_dict(torch.load("data/Witness_of_Babel.pth"))

noisegen = noisemaker.NoiseGen()
count = 0
start = time.time()

channel = bot.get_channel(os.getenv('CHANNEL_ID'))
@bot.event
async def on_ready():
    print("ready")

    while True:
        chunk = noisegen.generate_chunk(1000).to(device)
        outputs = witness(chunk)
        
        for idx, prediction in enumerate(outputs):
            if prediction[0] > 0.3:
                example = chunk[0].cpu()
                plt.imshow(example.reshape(64, 64, 1))
                plt.show()
                
                success = chunk[idx].cpu()
                plt.imshow(success.reshape(64, 64, 1))
                plt.show()
        
        count += 1000
    # _, truth = torch.max(labels, dim=1)

load_dotenv()
TOKEN = os.getenv('TOKEN')

bot.run(TOKEN)