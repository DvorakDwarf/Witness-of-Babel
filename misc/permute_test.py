#Trying to figure out how to go pixel-by-pixel across the Canvas

import itertools

#I got into my own head and thought it would be harder
#Found out I need combinations instead

stuff = [1, 2, 3]
for subset in itertools.product(stuff, repeat=5):
    print(subset)
