#Self-descriptive name. Used to view how my noise looks like

import torch
import torchvision
from matplotlib import pyplot as plt

while True:
    x = torch.rand(1, 24, 24)

    plt.imshow(x.reshape(24, 24, 1))
    plt.show()

    torchvision.utils.save_image(x, "data/noise_sample.png")