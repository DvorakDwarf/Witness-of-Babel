import torchvision

import logging
import os
import sys
import pickle
import datetime
import discord

class Logger:
    def __init__(self, channel, user_id):
        self.channel = channel
        self.user_id = user_id
        self.count = 0
        self.buffer = 0

        now=datetime.datetime.now()
        self.log_folder = f"logs/{now}"
        os.makedirs(self.log_folder)

        with open("logs/count.pickle", "rb") as f:
            self.count = pickle.load(f)

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

        with open(image_path, "rb") as fh:
            f = discord.File(fh, filename=image_path)

        await self.channel.send(message, file=f)

    async def log_anomalies(self, chunk, outputs):
        for idx, prediction in enumerate(outputs):
            real_certainty = prediction[0]
            match real_certainty:
                case real_certainty if real_certainty > 0.50:
                    image_path = self.save_image(chunk[idx], "BINGO")
                    message = f"<@{self.user_id}> BINGO at image: {self.count:02x} !"

                    await self.log(message, image_path)
                    break

                case real_certainty if real_certainty > 0.40:
                    image_path = self.save_image(chunk[idx], "40%")
                    message = f"<@{self.user_id}> Almost there at image: {self.count:02x} !"

                    await self.log(message, image_path)
                    break

                case real_certainty if real_certainty > 0.30:
                    breaimage_path = self.save_image(chunk[idx], "30%")
                    message = f"30% at image: {self.count:02x} !"

                    await self.log(message, image_path)
                    break

                case real_certainty if real_certainty > 0.25:
                    image_path = self.save_image(chunk[idx], "25%")
                    message = f"25% at image: {self.count:02x} !"

                    await self.log(message, image_path)
                    break

        self.count += len(chunk)
        self.buffer += len(chunk)

        print(self.count)
        if self.buffer > 1000000:
            with open("logs/count.pickle", "wb") as f:
                f.write(pickle.dumps(self.count))

            self.buffer = 0