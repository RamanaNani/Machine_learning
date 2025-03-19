import numpy as np

class SVM:
    def __init__(self, learning_rate = 0.001, lambda_ = 0.01, iters = 1000):
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.iters = iters
        self.w = None # Weight vector
        self.b = None # bias term

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # convert labels to -1 or 1 for SVM
        y_ = np.where(y <= 0, -1, 1)    

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx]*(np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_ * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_ * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)
    
    