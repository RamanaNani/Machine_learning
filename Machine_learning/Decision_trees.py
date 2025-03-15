import numpy as np

class DecisionTrees:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.tree = {}

    def entropy(self, y):
        hist = np.bincount(y)
        ps = hist/len(y)
        return -np.sum([p*np.log2(p) for p in ps if p > 0])
    
    def gain(self, X, y, feature, threshold):
        parent_entropy = self.entropy(y)
        left_indices, right_indices = X[:, feature] < threshold, X[:, feature] >= threshold
        if len(left_indices) == 0 or len(y[right_indices]) == 0:
            return 0
        n = len(y)