import torch

import noisemaker
from architecture import Witness

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
while True:
    chunk = noisegen.generate_chunk(1000)
    outputs = witness(chunk)

    print(outputs[0])

