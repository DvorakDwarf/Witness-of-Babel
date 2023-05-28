import torch
from matplotlib import pyplot as plt

while True:
    x = torch.rand(64, 64, 1)

    plt.imshow(x)
    plt.show()