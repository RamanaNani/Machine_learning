import numpy as np

class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)  # Find the unique class labels
        n_classes = len(self.classes)
        self.means = np.zeros((n_classes, n_features))
        self.variances = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.means[idx, :] = X_c.mean(axis=0)        # Mean per feature
            self.variances[idx, :] = X_c.var(axis=0)     # Variance per feature
            self.priors[idx] = X_c.shape[0] / n_samples  # Prior probability of class

    def predict(self, X):
        # Predict the class for each sample
        y_pred = [self._predict_single(x) for x in X]
        return np.array(y_pred)

    def _predict_single(self, x):
        # Calculate posterior probability for each class
        posteriors = []

        for idx, c in enumerate(self.classes):
            # Start with the log of the prior
            prior_log = np.log(self.priors[idx])
            # Add sum of log likelihoods of each feature
            likelihood_log = np.sum(self._log_gaussian_probability(idx, x))
            posterior = prior_log + likelihood_log
            posteriors.append(posterior)

        # Choose the class with the highest posterior probability
        return self.classes[np.argmax(posteriors)]

    def _log_gaussian_probability(self, class_idx, x):
        # Calculate log probability for Gaussian distribution:
        mean = self.means[class_idx]
        var = self.variances[class_idx]
        # Gaussian PDF (in log form to avoid underflow):
        return -0.5 * np.log(2 * np.pi * var) - ((x - mean) ** 2) / (2 * var)