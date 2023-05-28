import pickle

with open("logs/count.pickle", "wb") as f:
    f.write(pickle.dumps(1800000))