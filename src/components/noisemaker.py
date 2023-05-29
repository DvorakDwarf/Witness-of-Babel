#Class to make up noise in chunks
#Some functionality to go pixel-by-pixel. Inefficient

import torch

import numpy as np
from matplotlib import pyplot as plt
from collections import namedtuple

class NoiseGen:
    def __init__(self, img_size):
        
        self.img_size = img_size
    
    #What you should realistically use
    def generate_chunk(self, chunk_size):
        size = (chunk_size, 1, self.img_size, self.img_size)
        noise_chunk = torch.rand(size)

        return noise_chunk

    #SLOOOOOOOOOOOOOOOOOOW, but completely accurate
    def initialize(self, flat_noise, start):
        for _ in range(0, start):
            self.step(flat_noise)

    def step(self, flat_noise):
        for idx, pixel in enumerate(flat_noise):
            if idx == len(flat_noise)-1 and pixel >= 255:
                    #Congrats, you are at the end of canvas of babel !
                    flat_noise = np.zeros(self.img_size*self.img_size,)
                    break

            #in case added number to already 255 pixel
            if pixel > 255:
                flat_noise[idx] = 1
                flat_noise[idx+1] += 1
            elif idx == 0:
                flat_noise[idx] += 1

    #Option for snobs. Implement if you want
    def generate_specific_chunk(self, chunk_size, start=0):
        shape = (self.img_size, self.img_size, 1)
        noise = np.zeros(shape)

        flat_noise = noise.ravel()
        self.initialize(flat_noise, start)

        chunk = np.empty((chunk_size, 1, self.img_size, self.img_size))

        for chunk_idx in range(0, chunk_size):
            self.step(flat_noise)
            
            new_img = flat_noise.reshape(shape)
            chunk[chunk_idx] = new_img

        return chunk
        
