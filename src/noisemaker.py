import numpy as np
from matplotlib import pyplot as plt
from collections import namedtuple

Target = namedtuple("target", "x y brightness")

class NoiseGen:
    def __init__(self, log_path=None):

        self.log_path = log_path
    
    #Multithread would be good
    def generate_chunk(self):
        pass

    def generate_noise(self):
        noise = np.zeros((64, 64, 1))

        brightness = 0
        for x in range(0, 64):
            print(brightness)
            for y in range(0, 64):
                noise[x, y, 0] = brightness

            brightness += 1

        plt.imshow(noise)
        plt.show()

    def iterate_noise(self):
        noise = np.zeros((64, 64, 1))
        target = Target(0, 0, 1)
        affected = []

        noise[0, target.x, target.y] = target.brightness
        
        brightness = 0
        while True:
            for x in range(0, 64):
                for y in range(0, 64):
                    if x == target.x and y == target.y:
                        continue

                    noise[x, y, 0] = brightness

            brightness += 1

            if brightness > 255:
                brightness = 0
                
            plt.imshow(noise)
            plt.show()

    def better_iterate(self):
        noise = np.zeros((64, 64, 1))
        target = Target(0, 0, 1)
        affected = []

        noise[0, target.x, target.y] = target.brightness

        for x in range(0, 64):
            for y in range(0, 64):
                if x == target.x and y == target.y:
                    continue
                else:
                    affected.append([x, y])

        for x, y in affected:
            if noise[x, y, 0] == 255:
                continue
            else:
                noise[x, y, 0] += 1

        plt.imshow(noise)
        plt.show()



noisegen = NoiseGen()
noisegen.better_iterate()
        
