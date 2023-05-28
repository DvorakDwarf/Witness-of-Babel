import torch

import discord
from discord.ext import commands
import requests
import os
from dotenv import load_dotenv
import time
import asyncio

from components import noisemaker
from components.small_architecture import SmallWitness
from components import logger

#Reduce number if GPU too small
CHUNK_SIZE = 5000

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

witness = SmallWitness().to(device)
witness.load_state_dict(torch.load("data/Small_Witness_of_Babel.pth"))
noisegen = noisemaker.NoiseGen()
start = time.time()

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
        
        await HQ.log_anomalies(chunk, outputs)
        await asyncio.sleep(0.0001) #Heartbeat stops without this

TOKEN = os.getenv('TOKEN')

bot.run(TOKEN)