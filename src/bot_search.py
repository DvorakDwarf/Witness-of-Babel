#My primary search script
#Relies on correct info in .env
#Uses a discord bot for notifications

import torch

import discord
from discord.ext import commands
import requests
import os
from matplotlib import pyplot as plt
from dotenv import load_dotenv
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

witness = medium.MediumWitness().to(device)
witness.load_state_dict(torch.load("data/Medium_Witness_of_Babel_24.pth"))
noisegen = noisemaker.NoiseGen(IMAGE_SIZE)

load_dotenv()

@bot.event
async def on_ready():
    channel = bot.get_channel(int(os.getenv('CHANNEL_ID')))
    user_id = os.getenv('USER_ID')
    HQ = logger.Logger(channel, user_id)

    print("ready")

    while True:
        chunk = noisegen.generate_chunk(CHUNK_SIZE).to(device)
        outputs = witness(chunk)

        # #Uncomment to visualize data
        # plt.imshow(chunk[0].cpu().reshape(32, 32, 1))
        # plt.show()
        
        await HQ.log_anomalies(chunk, outputs)
        #Bot's heartbeat stops without this, disconnects
        await asyncio.sleep(0.0001) 

TOKEN = os.getenv('TOKEN')
bot.run(TOKEN)