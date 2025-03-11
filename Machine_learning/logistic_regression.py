import numpy as np

# Sigmoid function
def sigmoid(z):
    """Compute the sigmoid of z."""
    return 1 / (1 + np.exp(-z))

# Cost function (log loss)
def cost(y_pred, y, m):
    """Compute the log loss."""
    j = -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)) / m
    return j