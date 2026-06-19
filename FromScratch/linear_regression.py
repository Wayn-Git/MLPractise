import numpy as np

""" Formula for linear regression: y = mx + b
Where m = weight and b = bais 
y = dependent variable
x = independent varible

formula for m = sum(x - mean(x) * (y - mean(y)))

"""
class LinearRegression:
    def __init__(self, x, y, lr=0.1, w=0, b=0):
        self.x =  x
        self.y = y

x = np.random.randint(10)
print(x)