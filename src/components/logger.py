#Record all notable images and keep logs

import torchvision

import logging
import os
import sys
import pickle
import datetime
import discord

#Thresholds for when to log an image and when to ping
HI_THRESHOLD = 0.5
LO_THRESHOLD = 0.31

#How many images to go through before saving to count.pickle
BUFFER_SIZE = 1000000

class Logger:
    def __init__(self, channel=None, user_id=None, bot=True):
        self.channel = channel
        self.user_id = user_id
        self.bot = bot
        self.count = 0
        self.buffer = 0

        now=datetime.datetime.now()
        self.log_folder = f"logs/{now}"
        os.makedirs(self.log_folder)

        try:
            with open("logs/count.pickle", "rb") as f:
                self.count = pickle.load(f)
        except:
            with open("logs/count.pickle", "wb") as f:
                f.write(pickle.dumps(self.count))

        logging.basicConfig(
            level=logging.CRITICAL,
            filename=f"{self.log_folder}/records.log",
            filemode="w",
            format="%(levelname)s - %(asctime)s\n%(message)s \n"
        )
        logging.info("Logging has started")

    def save_image(self, image_tensor, name):
        # now=datetime.datetime.now()
        image_path = f"{self.log_folder}/{name}_{self.count:02x}.png"
        torchvision.utils.save_image(image_tensor, image_path)

        return image_path

    async def log(self, message, image_path):
        logging.critical(message)

        if self.bot == True:
            with open(image_path, "rb") as fh:
                f = discord.File(fh, filename=image_path)

            await self.channel.send(message, file=f)

    async def log_anomalies(self, chunk, outputs):
        for idx, prediction in enumerate(outputs):
            real_certainty = prediction[0]

            if real_certainty >= HI_THRESHOLD:
                image_path = self.save_image(chunk[idx], "BINGO")
                message = f"<@{self.user_id}> BINGO at image: {self.count:02x} ! ({real_certainty*100:.2f}%)"

                await self.log(message, image_path)

            elif real_certainty >= LO_THRESHOLD:
                image_path = self.save_image(chunk[idx], "Mid")
                message = f"Mid image: {self.count:02x} ({real_certainty*100:.2f}%)"

                await self.log(message, image_path)

        self.count += len(chunk)
        self.buffer += len(chunk)

        # print(self.count)
        if self.buffer > BUFFER_SIZE:
            with open("logs/count.pickle", "wb") as f:
                f.write(pickle.dumps(self.count))

            self.buffer = 0