# Libraries
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Algorithms
from hierarchical import AgglomerativeClustering as ac
from kmeans import KMeansAlg
from dbscan import DbscanClustering as dc

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

print("== DBSCAN with self-created algorithm ==")
dbs = dc(epsilon=0.7, min_pts=9, distance_metric='euclidean')
dbs.count_distance(X_test, X_train)
dbs.fit(X_train)
y_pred_dbs_self = dbs.predict(X_test)
print("Accuracy Score: " + str(accuracy_score(y_test, y_pred_dbs_self)))
print("")

print("== DBSCAN with scikit-learn library ==")
dbs_sklearn = DBSCAN(eps=0.7, min_samples=9, metric='euclidean')
dbs_sklearn.fit(X_train)
y_pred_dbs_sklearn = dbs_sklearn.labels_
print("Accuracy Score: " + str(accuracy_score(y_train, y_pred_dbs_sklearn)))
print("")

print("== Agglomerative with self-created algorithm ==")
agglo = ac(n_clusters=3, linkage='single', affinity='euclidean')
agglo.fit(X_train)
y_pred_agglo_self = agglo.predict(X_test)
print("Accuracy Score: " + str(accuracy_score(y_test, y_pred_agglo_self)))
print("")

print("== Agglomerative with scikit-learn library ==")
agglo_sklearn = AgglomerativeClustering(n_clusters=3, linkage='single', affinity='euclidean')
agglo_sklearn.fit(X_train)
y_pred_agglo_sklearn = agglo_sklearn.labels_
print("Accuracy Score: " + str(accuracy_score(y_train, y_pred_agglo_sklearn)))
print("")

print("== KMeans with self-created algorithm ==")
kmeans_self = KMeansAlg(n_clusters=3)
kmeans_self.fit(X_train)
y_pred_kmeans_self = kmeans_self.predict(X_test)
print("Accuracy Score: " + str(accuracy_score(y_test, y_pred_kmeans_self)))
print("")

print("== KMeans with scikit-learn library ==")
kmeans_sklearn = KMeans(n_clusters=3)
kmeans_sklearn.fit(X_train)
y_pred_kmeans_sklearn = kmeans_sklearn.predict(X_test)
print("Accuracy Score: " + str(accuracy_score(y_test, y_pred_kmeans_sklearn)))
print("")