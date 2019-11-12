import numpy as numpy
from scipy import spatial

def dbscan(dataset, epsilon, min_pts, distance_metric = 'euclidean'):
	nrow, ncolumn = numpy.shape(dataset)
	visited = numpy.zeros(nrow, 'int')
	point_type = numpy.zeros(nrow)	# -1: noise, 0: border, 1: core point
	
	neighbors = [[] for i in range(nrow)]	#list of neighbors of each point
	distance_matrix = spatial.distance.squareform(spatial.distance.pdist(dataset, distance_metric))

	# point_cluster_number = numpy.zeros(nrow)
	clusters_list = []	#list of clusters

	#assign point_type
	for point in range(nrow):
		neighbors[point] = numpy.nonzero(distance_matrix[point] < epsilon)[0]
		# neighbors[point].remove(point)	#remove point from its neighbor
		if len(neighbors[point]) < min_pts:
			point_type[point] = -1	#assign first as outlier
		else:
			point_type[point] = 1	#assign as core point
		neighbors[point].tolist().remove(point)	#remove point from its neighbor

	#clustering based on point_type
	for point in range(nrow):
		unique_cluster = []
		cluster = []	#list of points in a cluster
		if visited[point] == 0:
			visited[point] = 1
			if point_type[point] == 1:
				cluster.append(point)
				cluster.extend(neighbors[point])
				cluster_neighbors(neighbors, point, cluster, point_type, visited)
			
				# Save the cluster
				unique_cluster = list(set(cluster))
				clusters_list.append(unique_cluster)
	
	return clusters_list

def cluster_neighbors(neighbors, point, cluster, point_type, visited):	#recursive procedure
	for p in neighbors[point]:
		if visited[p] == 0:
			visited[p] = 1
			if point_type[p] == 1:
				cluster.append(p)
				cluster.extend(neighbors[p])
				cluster_neighbors(neighbors, p, cluster, point_type, visited)
			
					

dataset = [[0,0], [3,4], [1,1], [3,3]]
epsilon = 2
minpts = 2
clust = dbscan(dataset, epsilon, minpts)
print(clust)