import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

""" Formula for linear regression: y = mx + b
Where m = weight and b = bais 
y = dependent variable
x = independent varible

formula for m = sum(x - mean(x) * (y - mean(y)))

"""

X = np.array([1, 2, 3, 4, 5], dtype=float)

y = np.array([2, 4, 6, 8, 10], dtype=float)


class LinearRegression:
    def __init__(self, lr_rate, epochs):
        self.lr_rate = lr_rate
        self.epochs = epochs
        self.w = None
        self.b = None
    #https://machinelearningmastery.com/linear-regression-for-machine-learning/
    def predict(self, X):
        return X * self.w + self. b
    
    # https://www.geeksforgeeks.org/python/python-mean-squared-error/
    def MSE(self, y_pred, y_true):

        return np.mean(y_true - y_pred)**2

    # yi = y_true, m = w, b = b, xi = X
    # https://www.geeksforgeeks.org/machine-learning/gradient-descent-in-linear-regression/
    def gradient(self, X, y, loss):

        wg = -2 * (np.mean(X *(y - (self.w * X + self.b)))) 
        wb = -2 * (np.mean((y - (self.w * X + self.b)))) 

        return wg, wb
    
    def fit(self, X, y):

        self.w = np.random.randn(1, 1)
        self.b = np.random.randn(1)
        for epoch in range(self.epochs):
            y_pred = self.predict(X)

            loss = self.MSE(y_pred, y)

            wg, wb = self.gradient(X, y, loss)

            self.w -= self.lr_rate * wg
            self.b -= self.lr_rate * wb
            
            if epoch % 100 == 0:
                print(f"epoch: {epoch}, Loss: {loss:.6f}")




       
lr_model = LinearRegression(0.01, 1000)

lr_model.fit(X, y)

