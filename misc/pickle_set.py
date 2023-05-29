#I never used pickle before this, shush
#Also used it to reset value of count.pickle
#count.pickle contains the number of images the AI went through

import pickle

with open("logs/count.pickle", "wb") as f:
    f.write(pickle.dumps(1800000))