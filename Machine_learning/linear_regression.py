import numpy as np
class LinearRegression:
    def __init__(self, learning_rate = 0.01, max_iter = 1000, lambda_ = 0.1, regularization = 'l2'):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.lambda_ = lambda_
        self.regularization = regularization

    def mse(self, y_pred, y, m):
        # Calculate Mean Squared Error to calculate the loss
        j = np.sum((y_pred - y)**2)/(2*m) 
        return j
    
    def fit(self, X, y):
        m, n = X.shape
        X = np.hstack([np.ones((m,1)), X])
        self.w = np.zeros(n+1)

        for i in range(self.max_iter):
            y_pred = X.dot(self.w)
            gradient = (1 / m) * X.T.dot(y_pred - y) # calculate gradients

            if self.regularization == 'l2':
                penalty = (self.lambda_ / m) * self.w[1:]  # Exclude bias term after the derivative
                gradient[1:] += penalty
            elif self.regularization == 'l1':
                penalty = (self.lambda_ / m) * np.sign(self.w[1:])  # Exclude bias term
                gradient[1:] += penalty

            self.w -= self.learning_rate * gradient # Update weights

        
    def predict(self, X):
        # Make predictions using the trained model.
        m = X.shape[0]
        X = np.hstack([np.ones((m, 1)), X])  # Add bias term
        y_pred = X.dot(self.w)
        return y_pred
        