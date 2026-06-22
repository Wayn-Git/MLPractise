import numpy as np

""" Formula for linear regression: y = mx + b
Where m = weight and b = bais 
y = dependent variable
x = independent varible

formula for m = sum(x - mean(x) * (y - mean(y)))

"""

# The number of training data
N = 200
# 200 random samples as our data
x_1 = np.random.rand(N)
# Define the line slope and the Gaussian noise parameters
slope = 3
mu, sigma = 0, 0.1 # mean and standard deviation
intercept = np.random.normal(mu, sigma, N)
# Define the coordinates of the data points using the line equation and the added Gaussian noise 
y = slope*x_1 + intercept

class LinearRegression:
    def __init__(self, x, y, lr=0.1, w=0, b=0):
        self.x =  x
        self.y = y
    
    def MSE(self, y_pred):
        return np.mean((self.y - y_pred) ** 2)
    
    
x = np.random.randint(10)
print(x)