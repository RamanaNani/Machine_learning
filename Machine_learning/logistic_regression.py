import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, lambda_=0.1, regularization='l2', max_iter=1000):
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.regularization = regularization
        self.max_iter = max_iter

    # Sigmoid function
    def sigmoid(self, z):
        """Compute the sigmoid of z."""
        return 1 / (1 + np.exp(-z))

    # Cost function (log loss) to calculate the loss
    def loss(self, y_pred, y, m):
        """Compute the log loss."""
        j = -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)) / m
        return j
    #After derivation the value is dl/dw = -1/m(y-y_hat)X

    # Fit function is used to fit the model
    def fit(self, X, y):
        m, n = X.shape
        X = np.hstack([np.ones((m, 1)), X])  # Add bias term
        self.w = np.zeros(n + 1)

        for i in range(self.max_iter):
            z = X.dot(self.w)
            h = self.sigmoid(z)
            gradient = (1 / m) * X.T.dot(h - y)

            if self.regularization == 'l2':
                penalty = (self.lambda_ / m) * self.w[1:]  # Exclude bias term after the derivative
                gradient[1:] += penalty
            elif self.regularization == 'l1':
                penalty = (self.lambda_ / m) * np.sign(self.w[1:])  # Exclude bias term
                gradient[1:] += penalty  # Add penalty to gradient
            self.w -= self.learning_rate * gradient  # Correct weight update

    # Prediction function
    def predict(self, X):
        # Make predictions using the trained model.
        m = X.shape[0]
        X = np.hstack([np.ones((m, 1)), X])  # Add bias term
        z = X.dot(self.w)
        y_pred = self.sigmoid(z)
        return np.round(y_pred)