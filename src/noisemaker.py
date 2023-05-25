import torch

import numpy as np
from matplotlib import pyplot as plt
from collections import namedtuple

class NoiseGen:
    def __init__(self, log_path=None):

        self.log_path = log_path
    
    #What you should realistically use
    def generate_chunk(self, chunk_size):
        size = (chunk_size, 64, 64, 1)
        noise_chunk = torch.rand(size)

        return noise_chunk

    def generate_specific_chunk(self, chunk_size, start=0):
        shape = (64, 64, 1)
        noise = np.zeros(shape)

        flat_noise = noise.ravel()
        self.initialize(flat_noise, start)

        chunk = np.empty((chunk_size, 64, 64, 1))

        for chunk_idx in range(0, chunk_size):
            self.step(flat_noise)
            
            new_img = flat_noise.reshape(shape)
            chunk[chunk_idx] = new_img

        return chunk

    def step(self, flat_noise):
        for idx, pixel in enumerate(flat_noise):
            if idx == len(flat_noise)-1 and pixel >= 255:
                    #Congrats, you are at the end of canvas of babel !
                    flat_noise = np.zeros(4096,)
                    break

            #in case added number to already 255 pixel
            if pixel > 255:
                flat_noise[idx] = 1
                flat_noise[idx+1] += 1
            elif idx == 0:
                flat_noise[idx] += 1

    #SLOOOOOOOOOOOOOOOOOOW, but completely accurate
    def initialize(self, flat_noise, start):
        for _ in range(0, start):
            print(_)
            self.step(flat_noise)


noisegen = NoiseGen()
chunk = noisegen.generate_chunk(1000)

plt.imshow(chunk[999])
plt.show()

        
