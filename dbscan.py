import numpy as numpy
from scipy import spatial

def dbscan(dataset, epsilon, min_pts, distance_metric = 'euclidean'):
	nrow, ncolumn = dataset.shape
	visited = numpy.zeros(nrow, 'int')
	point_type = numpy.zeros(nrow)	# -1: noise, 0: border, 1: core point
	
	neighbors = numpy.zeros(nrow)	#list of neighbors of each point
	distance_matrix = spatial.distance.squareform(spatial.distance.pdist(dataset, distance_metric))

	# point_cluster_number = numpy.zeros(nrow)
	clusters_list = []	#list of clusters

	#assign point_type
	for point in range(nrow):
		neighbors[point] = numpy.nonzero(distance_matrix[point] < epsilon)[0]
		neighbors[point].remove(point)	#remove point from its neighbor
		if len(neighbors[point]) < min_pts:
			point_type[point] = -1	#assign first as outlier
		else:
			point_type[point] = 1	#assign as core point

	#clustering based on point_type
	for point in range(nrow):
		unique_cluster = []
		cluster = []	#list of points in a cluster
		if visited[point] == 0:
			if point_type[point] == 1:
				cluster.extend(point)
				cluster.extend(neighbors[point])
				cluster_neighbors(neighbors[point], cluster, point_type, visited)
			
			# Save the cluster
			unique_cluster = list(set(cluster))
			clusters_list.append(unique_cluster)

			visited[point] = 1
	
	return clusters_list

def cluster_neighbors(neighbors, cluster, point_type, visited):	#recursive procedure
	for point in neighbors:
		if visited[point] == 0:
			if point_type[point] == 1:
				cluster.extend(point)
				cluster.extend(neighbors[point])
				cluster_neighbors(neighbors[point], cluster, point_type, visited)
			visited[point] = 1
					


# def set2List(NumpyArray):
# 	list = []
# 	for item in NumpyArray:
# 		list.append(item.tolist())
# 	return list

# def DBSCAN(Dataset, Epsilon, MinPts, DistanceMethod = 'euclidean'):
# 	#    Dataset is a mxn matrix, m is number of item and n is the dimension of data
# 	m,n = Dataset.shape
# 	Visited=numpy.zeros(m,'int')
# 	Type=numpy.zeros(m)
# 	#   -1 noise, outlier
# 	#    0 border
# 	#    1 core
# 	ClustersList=[]
# 	Cluster=[]
# 	PointClusterNumber=numpy.zeros(m)
# 	PointClusterNumberIndex=1
# 	PointNeighbors=[]
# 	DistanceMatrix = spatial.distance.squareform(spatial.distance.pdist(Dataset, DistanceMethod))
# 	for i in range(m):
# 		if Visited[i]==0:
# 			Visited[i]=1
# 			PointNeighbors=numpy.nonzero(DistanceMatrix[i]<Epsilon)[0]
# 			if len(PointNeighbors)<MinPts:
# 				Type[i]=-1
# 			else:
# 				for k in range(len(Cluster)):
# 					Cluster.pop()
# 				Cluster.append(i)
# 				PointClusterNumber[i]=PointClusterNumberIndex
				
				
# 				PointNeighbors=set2List(PointNeighbors)    
# 				ExpandClsuter(Dataset[i], PointNeighbors,Cluster,MinPts,Epsilon,Visited,DistanceMatrix,PointClusterNumber,PointClusterNumberIndex  )
# 				Cluster.append(PointNeighbors[:])
# 				ClustersList.append(Cluster[:])
# 				PointClusterNumberIndex=PointClusterNumberIndex+1	
# 	return PointClusterNumber



# Dataset = [[0,0], [1,1], [3,4], [3,3]]
# DistanceMatrix = spatial.distance.squareform(spatial.distance.pdist(Dataset))
# print(DistanceMatrix)

# # print(numpy.nonzero(DistanceMatrix[1]<5)[0])

# # PointNeighbors=numpy.asarray(DistanceMatrix[0]<5).nonzero()[0]
# PointNeighbors=numpy.where(DistanceMatrix[2]<5)[0]
# print(PointNeighbors)