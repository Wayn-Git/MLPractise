import numpy as np


# -------------------------
# Layers
# -------------------------

def linear_forward(x, W, b):
    return x @ W + b


def linear_backward(x, W, grad):
    dW = x.T @ grad
    db = np.sum(grad, axis=0, keepdims=True)
    dx = grad @ W.T

    return dW, db, dx


def relu_forward(x):
    return np.maximum(x, 0)


def relu_backward(x, grad):
    return grad * (x > 0)


# -------------------------
# Data
# -------------------------

x = np.arange(1, 100, 3).reshape(-1, 1)
y = x ** 2 + 6


# -------------------------
# Parameters
# -------------------------

np.random.seed(0)

W0 = np.random.randn(1, 16) * 0.01
b0 = np.zeros((1, 16))

W1 = np.random.randn(16, 16) * 0.01
b1 = np.zeros((1, 16))

W2 = np.random.randn(16, 1) * 0.01
b2 = np.zeros((1, 1))


learning_rate = 0.000001
epochs = 5000

losses = []


# -------------------------
# Training
# -------------------------

for epoch in range(epochs):

    # Forward

    L0 = linear_forward(x, W0, b0)
    R0 = relu_forward(L0)

    L1 = linear_forward(R0, W1, b1)
    R1 = relu_forward(L1)

    L2 = linear_forward(R1, W2, b2)

    loss = np.mean((L2 - y) ** 2)
    losses.append(loss)


    # Backward

    dL2 = (2 / x.shape[0]) * (L2 - y)

    dW2, db2, dR1 = linear_backward(R1, W2, dL2)

    dL1 = relu_backward(L1, dR1)

    dW1, db1, dR0 = linear_backward(R0, W1, dL1)

    dL0 = relu_backward(L0, dR0)

    dW0, db0, dx = linear_backward(x, W0, dL0)


    # Update

    W0 -= learning_rate * dW0
    b0 -= learning_rate * db0

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2