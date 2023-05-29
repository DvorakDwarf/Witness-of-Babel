#I wanted to make sure images going through .ravel() come back correct

import numpy as np
from PIL import Image

img = Image.open('data/index.png').convert('RGB')
arr = np.array(img)

# record the original shape
shape = arr.shape
print(shape)

# make a 1-dimensional view of arr
flat_arr = arr.ravel()
print(flat_arr.shape)

# reform a numpy array of the original shape
arr2 = flat_arr.reshape(shape)

# make a PIL image
img2 = Image.fromarray(arr2, 'RGB')
img2.show()