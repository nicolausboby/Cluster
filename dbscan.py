import numpy as np
from scipy import spatial

class DbscanClustering():

	def __init__(self, epsilon, min_pts, distance_metric = 'euclidean'):
		print("Dbscan Model initiated")
		self.epsilon = epsilon
		self.min_pts = min_pts
		self.distance_metric = distance_metric
		self.clusters_list = []		# list of clusters
		self.dist_mat = []			# distance matrix between each point in datatest to each point in dataset
	
	def fit(self, dataset):
		self.clusters_list = self._dbscan(dataset, self.epsilon, self.min_pts, self.distance_metric)

	def count_distance(self, datatest, dataset):
		# count distance between each point in datatest to each point in dataset
		# return a list of list of distance per point in datatest
		dist_mat = []
		for p in datatest:
			temp = []
			for q in dataset:
				temp.append(spatial.distance.euclidean(p,q))
			dist_mat.append(temp)
		self.dist_mat = dist_mat

	def predict(self, datatest):
		# predict cluster based on dist_mat
		# returns a list of labels
		clusters = []
		nrow = np.shape(datatest)[0]
		for point in range(nrow):
			min_distance = min(self.dist_mat[point])
			nearest_point = self.dist_mat[point].index(min_distance)
			found = False
			for c in self.clusters_list:
				for p in c:
					if(nearest_point == p):
						clusters.append(self.clusters_list.index(c))
						found = True
			if(found == False):
				clusters.append(-1)	# assign as outlier

		return clusters

	def _dbscan(self, dataset, epsilon, min_pts, distance_metric):
		nrow, ncolumn = np.shape(dataset)
		visited = np.zeros(nrow, 'int')
		point_type = np.zeros(nrow)				# -1: noise, 0: border, 1: core point
		neighbors = [[] for i in range(nrow)]	# list of neighbors of each point
		distance_matrix = spatial.distance.squareform(spatial.distance.pdist(dataset, distance_metric))
		clusters_list = []						# list of clusters

		# assign point_type
		for point in range(nrow):
			neighbors[point] = np.nonzero(distance_matrix[point] <= epsilon)[0]
			if len(neighbors[point]) <= min_pts:
				point_type[point] = -1			# assign first as outlier
			else:
				point_type[point] = 1			# assign as core point
			neighbors[point].tolist().remove(point)	# remove point from its neighbor

		# clustering based on point_type
		for point in range(nrow):
			unique_cluster = []
			cluster = []						# list of points in a cluster
			if visited[point] == 0:
				visited[point] = 1
				if point_type[point] == 1:
					cluster.append(point)
					cluster.extend(neighbors[point])
					self.cluster_neighbors(neighbors, point, cluster, point_type, visited)
					unique_cluster = list(set(cluster))		# delete duplicate members
					clusters_list.append(unique_cluster)	# save the cluster
		
		for cluster in clusters_list:
			for point in cluster:
				if point_type[point] == -1:
					point_type[point] = 0	# assign as border point

		return clusters_list
	
	def cluster_neighbors(self, neighbors, point, cluster, point_type, visited):
		# recursive procedure to cluster neighbors
		for p in neighbors[point]:
			if visited[p] == 0:
				visited[p] = 1
				if point_type[p] == 1:
					cluster.append(p)
					cluster.extend(neighbors[p])
					self.cluster_neighbors(neighbors, p, cluster, point_type, visited)