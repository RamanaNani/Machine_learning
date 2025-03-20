import numpy as np

class KMeans:
    def __init__(self, k, max_iter = 1000, tol = 1e-4):
        self.k =k
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None

    def initial_centroids(self, X):
        np.random.seed(42)
        # intial get random centroids with in the dataset
        indices = np.random.choice(X.shape[0], self.k, replace = False)
        self.centroids = X[indices,:]

    def calculate_distance(self, X, centroids):
        # calculate the eculidean distance between points and centroids
        eculidean = np.sum((X[:,np.newaxis]-centroids)**2, axis = 2)
        return np.sqrt(eculidean)

    def get_clusters(self, X):
        # Assign each data point to the closet centroid
        distances = self.calculate_distance(X, self.centroids)
        return np.argmin(distances, axis = 1)
    
    def update_centroids(self, X, cluster):
        # Update centroids based on the the cluster values
        new_centroids = np.array([X[cluster == i].mean(axis = 0) for i in range(self.k)])
        return new_centroids

    def fit(self, X):
        self.initial_centroids(X)
        for i in range(self.max_iter):
            cluster = self.get_clusters(X)
            new_centroids = self.update_centroids(X,cluster)
            # if the distance between new centroid and previous is less than tol stop the training
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
            self.centroids = new_centroids

    def predict(self, X):
        # predict the cluster points
        return self.get_clusters(X)
    

if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(100, 2)

    kmeans = KMeans(k=3)
    kmeans.fit(X)

    cluster_assignments = kmeans.predict(X)
    import matplotlib.pyplot as plt
    plt.scatter(X[:, 0], X[:, 1], c=cluster_assignments, cmap='viridis')
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', s=200, alpha=0.5)
    plt.show()