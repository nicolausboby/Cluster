"""
Hierarchical clustering 

"""
import numpy as np
from scipy.spatial import distance

def agglomerative(data, N_cluster, linkage, calc_dist):
    """
    Calculate dissimilarity or distance between 2 clusters

    Parameters
    ----------
    data : list.
        Input data that will be clustered
    
    N_cluster : int.
        Number of output clusters

    linkage : str
        linkage type for calculating dissimilarity. 
        Valid modes : single, complete, average, average-group

    calc_dist : str
        Type for calculating distance. 
        Valid modes : euclidean, manhattan    

    Returns
    -------
    clusters : list
        Output clusters
    """
    if len(data) < N_cluster:
        raise ValueError('N_cluster can not be bigger than the number of data')
    elif len(data) == N_cluster or len(data) == 1:
        return data

    # Singleton
    clusters = [[data[i]] for i in range(len(data))]
    cluster_count = len(clusters)

    while cluster_count > N_cluster:
        closest_d = _linkage(clusters[0], clusters[1], calc_dist, linkage)
        closest_pair = (0, 1)
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                temp = _linkage(clusters[i], clusters[j], calc_dist, linkage)
                if temp < closest_d: 
                    closest_d = temp
                    closest_pair = (i, j)
        new_cluster = clusters[closest_pair[0]] + clusters[closest_pair[1]]
        clusters.pop(max(closest_pair[0], closest_pair[1]))
        clusters.pop(min(closest_pair[0], closest_pair[1]))
        clusters.append(new_cluster)
        cluster_count -= 1
    return clusters

def _linkage(c1, c2, d_mode, mode = 'single'):
    """
    Calculate dissimilarity or distance between 2 clusters

    Parameters
    ----------
    c1 : numpy.array or list
        cluster 1
    
    c2 : numpy.array or list
        cluster 2

    mode : str
        linkage type for calculating dissimilarity
        valid modes : single, complete, average, average-group

    d_mode : str
        type for calculating distance 
        valid modes : euclidean, manhattan    

    Returns
    -------
    d : float
        Dissimilarity result between 2 clusters, calculated based on linkage mode
    """
    mode = mode.lower()
    if mode ==  'single':
        min_p = c1[0]
        min_q = c2[0]
        min_d = _distance(min_p, min_q, d_mode)
        for p in c1:
            for q in c2:
                min_d = min(_distance(p, q, d_mode), min_d)
        return min_d

    elif mode == 'complete':
        max_p = c1[0]
        max_q = c2[0]
        max_d = _distance(max_p, max_q, d_mode)
        for p in c1:
            for q in c2:
                max_d = max(_distance(p, q, d_mode), max_d)
        return max_d

    elif mode == 'average-group':
        avgc1 = []
        avgc2 = []
        for axis in range(len(p[0])):
            sum_p = sum_q =  0
            for p in c1:
                sum_p += p[axis]
            avgc1.append(sum_p/len(p[0]))
            for q in c2:
                sum_q += q[axis]
            avgc2.append(sum_q/len(q[0]))

        return _distance(avgc1, avgc2, d_mode)

    elif mode == 'average':
        pass
    else:
        raise Exception('linkage mode {} is invalid'.format(mode))



def _distance(p1, p2, d_mode='euclidean'):
    """
    Calculate distance between 2 data

    Parameters
    ----------
    p1 : list
        data point 1
    
    p2 : list
        data point 2

    d_mode : str
        type for calculating distance 
        valid modes : euclidean, manhattan

    Returns
    -------
    d : float
        Distance between 2 data points
    """
    d_mode = d_mode.lower()
    if d_mode == 'euclidean':
        return distance.euclidean(p1, p2)


dataset = [[0,0], [3,4], [1,1], [3,3]]
clust = agglomerative(dataset, 3, 'single', 'euclidean')
print(clust)