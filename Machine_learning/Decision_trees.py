# import numpy as np

# class DecisionTrees:
#     def __init__(self, max_depth):
#         self.max_depth = max_depth
#         self.tree = {}

#     def entropy(self, y):
#         hist = np.bincount(y)
#         ps = hist/len(y)
#         return -np.sum([p*np.log2(p) for p in ps if p > 0])
    
#     def gain(self, X, y, feature, threshold):
#         parent_entropy = self.entropy(y)
#         left_indices, right_indices = X[:, feature] < threshold, X[:, feature] >= threshold
#         if len(left_indices) == 0 or len(y[right_indices]) == 0:
#             return 0
#         n = len(y)

import numpy as np

class DecisionTrees:
    def __init__(self, max_depth=5, min_samples_split=2):
        # Initialize the tree parameters
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    # Calculate entropy (measures impurity)
    def entropy(self, y):
        hist = np.bincount(y)            # Count occurrences of each class
        ps = hist / len(y)               # Convert to probabilities
        return -np.sum([p * np.log2(p) for p in ps if p > 0])  # Entropy formula

    # Calculate information gain for a given feature and threshold
    def gain(self, X, y, feature, threshold):
        parent_entropy = self.entropy(y)
        left_indices = X[:, feature] < threshold
        right_indices = X[:, feature] >= threshold

        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return 0   # No gain if one split is empty

        n = len(y)
        left_entropy = self.entropy(y[left_indices])
        right_entropy = self.entropy(y[right_indices])

        # Weighted entropy of children
        weighted_entropy = (np.sum(left_indices) / n) * left_entropy + \
                           (np.sum(right_indices) / n) * right_entropy

        return parent_entropy - weighted_entropy  # Gain = Parent entropy - Children entropy

    # Function to find the best feature and threshold to split
    def best_split(self, X, y):
        best_gain = 0
        best_feature = None
        best_threshold = None

        # Check each feature and each unique value as potential split
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                current_gain = self.gain(X, y, feature, threshold)
                if current_gain > best_gain:
                    best_gain = current_gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    # Recursive function to build the tree
    def build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        # Stop conditions: max depth reached or pure node or too few samples
        if (depth >= self.max_depth or num_labels == 1 or num_samples < self.min_samples_split):
            leaf_value = self._majority_class(y)
            return {'leaf': leaf_value}

        feature, threshold = self.best_split(X, y)

        # If no good split found, make a leaf
        if feature is None:
            leaf_value = self._majority_class(y)
            return {'leaf': leaf_value}

        # Split and continue building the tree recursively
        left_indices = X[:, feature] < threshold
        right_indices = X[:, feature] >= threshold

        left_subtree = self.build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self.build_tree(X[right_indices], y[right_indices], depth + 1)

        # Store node info
        return {
            'feature': feature,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree
        }

    # Fit function to start building the tree
    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    # Predict a single sample by traversing the tree
    def predict_sample(self, sample, node):
        if 'leaf' in node:
            return node['leaf']

        if sample[node['feature']] < node['threshold']:
            return self.predict_sample(sample, node['left'])
        else:
            return self.predict_sample(sample, node['right'])

    # Predict for a batch of samples
    def predict(self, X):
        return np.array([self.predict_sample(sample, self.tree) for sample in X])

    # Helper function to pick majority class for leaf nodes
    def _majority_class(self, y):
        counts = np.bincount(y)
        return np.argmax(counts)