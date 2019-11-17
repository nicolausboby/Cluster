# Libraries
import copy
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Helper Functions
def euclidean(a, b):
    return np.linalg.norm(a-b)

class KMeansAlg:

    def __init__(self, n_clusters=3):
        print("Kmeans Model initiated")
        self.num_clusters = n_clusters
        self.centroids = np.zeros(shape=(n_clusters, 1))

    def fit(self, input_dataset):

        # Set centroids randomly
        centroids = input_dataset[np.random.choice(input_dataset.shape[0], self.num_clusters, replace=False)]

        # Set list of list to store clusters
        clusters = []
        for i in range(self.num_clusters):
            clusters.append([])

        # KMeans Iteration
        error = 999999999999999

        while error != 0:
            # Get new clusters and centroids
            clusters_new = []
            for i in range(self.num_clusters):
                clusters_new.append([])

            centroids_new = copy.deepcopy(centroids)

            # Cluster data according to euclidean distances
            for i in range(input_dataset.shape[0]):
                distances = []
                for j in range(centroids.shape[0]):
                    distances.append(euclidean(input_dataset[i], centroids[j]))

                cluster_idx = distances.index(min(distances))
                clusters_new[cluster_idx].append(input_dataset[i])

            # Get means per cluster and assign new centroids
            mean = np.zeros(centroids.shape)
            for i in range(len(clusters)):
                for j in range(len(clusters[i])):
                    mean[i][j] = np.mean(input_dataset[i][j])
                centroids_new[i] = mean[i]

            error = euclidean(centroids, centroids_new)
            centroids = centroids_new

        print("Fit successful!")

    def predict(self, input_test):
        belongs_to = np.zeros(input_test.shape[0], dtype=np.int8)

        for i in range(input_test.shape[0]):
            distances = []
            for j in range(self.centroids.shape[0]):
                distances.append(euclidean(input_test[i], self.centroids[j]))

            cluster_idx = distances.index(min(distances))
            belongs_to[i] = cluster_idx

        return belongs_to

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

print("== Dataset Loading ==")
print("Initial X shape: " + str(X.shape))
print("Initial y shape: " + str(y.shape))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("X Train shape : " + str(X_train.shape))
print("X Test shape  : " + str(X_test.shape))
print("y Train shape : " + str(y_train.shape))
print("y Test shape  : " + str(y_test.shape))
print("")

print("== KMeans with self-created algorithm ==")
kmeans_alg = KMeansAlg(n_clusters=3)
kmeans_alg.fit(X_train)
y_pred_alg = kmeans_alg.predict(X_test)
print("Accuracy Score: " + str(accuracy_score(y_test, y_pred_alg)))
print("")

print("== KMeans with scikit-learn library ==")
kmeans_sklearn = KMeans(n_clusters=3)
kmeans_sklearn.fit(X_train)
y_pred_sklearn = kmeans_sklearn.predict(X_test)
print("Accuracy Score: " + str(accuracy_score(y_test, y_pred_sklearn)))
