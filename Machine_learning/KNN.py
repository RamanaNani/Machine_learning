import numpy as np

class KNN:
    def __init__(self, k=3):

        self.k = k

    def euclidean_distance(self, x1, x2):
        # calculate the euclidean distance bewtween points x1 and x2
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def get_neighbors(self, train_X, train_y, test_row):
        # calculate the distance from points to test point
        distances = [(row, label, self.euclidean_distance(row, test_row)) for row, label in zip(train_X, train_y)]
        # sort the points as per distances
        distances.sort(key=lambda x: x[2])
        return distances[:self.k]

    def predict(self, train_X, train_y, test_row):
        neighbors = self.get_neighbors(train_X, train_y, test_row)
        output_values = [neighbor[1] for neighbor in neighbors]
        # use voting and return max votes class
        prediction = max(set(output_values), key=output_values.count)
        return prediction


# Example usage
if __name__ == "__main__":
    # Sample dataset
    train_X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    train_y = np.array([0, 0, 0, 1, 1])
    
    # New data point
    test_row = np.array([3.5, 4.5])
    
    # Run KNN
    knn = KNN(k=3)
    prediction = knn.predict(train_X, train_y, test_row)
    
    print(f"Predicted class: {prediction}")
