# Libraries
import copy
import numpy as np
from sklearn import datasets
from collections import Counter
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

print("X shape: " + str(X.shape))
print("y shape: " + str(y.shape))
print("Number of cluster needed : " + str(len(Counter(y).keys())))

# Helper Functions
def euclidean(a, b):
    return np.linalg.norm(a-b)

# Kmeans Function
def kmeans(input_dataset, number_of_groups=3):
    # Set centroids
    centroids = input_dataset[np.random.choice(input_dataset.shape[0], number_of_groups, replace=False)]
    clusters = []
    for i in range(number_of_groups):
        clusters.append([])

    # KMeans Iteration
    error = 9999999999999999
    belongs_to = np.zeros(input_dataset.shape[0], dtype=np.int8)

    while error != 0:
        # Get new clusters and centroids
        clusters_new = []
        for i in range(number_of_groups):
            clusters_new.append([])

        centroids_new = copy.deepcopy(centroids)

        # Cluster data according to euclidean distances
        for i in range(input_dataset.shape[0]):
            distances = []
            for j in range(centroids.shape[0]):
                distances.append(euclidean(input_dataset[i], centroids[j]))

            cluster_idx = distances.index(min(distances))
            belongs_to[i] = cluster_idx
            clusters_new[cluster_idx].append(input_dataset)

        # Get means per cluster and assign new centroids
        mean = np.zeros(centroids.shape)
        for i in range(len(clusters)):
            for j in range(len(clusters[i])):
                mean[i][j] = np.mean(input_dataset[i][j])
            centroids_new[i] = mean[i]

        error = euclidean(centroids, centroids_new)
        centroids = centroids_new

        return belongs_to

# Examples
y_pred = kmeans(X, len(Counter(y).keys()))

print("Accuracy Score: " + str(accuracy_score(y, y_pred)))