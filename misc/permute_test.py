import itertools

#I got into my own head and thought it would be harder
#Found out I need combinations instead

stuff = [1, 2, 3]
for subset in itertools.product(stuff, repeat=5):
    print(subset)

# noise = itertools.product(range(0, 255 + 1), repeat=4096)
# print(len(noise))