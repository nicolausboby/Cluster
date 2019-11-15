"""
Hierarchical clustering 

"""
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage

def agglomerative():
    pass

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



def _distance(p1, p2, mode='euclidean'):
    """
    Calculate distance between 2 data

    Parameters
    ----------
    p1 : list
        data point 1
    
    p2 : list
        data point 2

    mode : str
        type for calculating distance 
        valid modes : euclidean, manhattan

    Returns
    -------
    d : float
        Distance between 2 data points
    """
    mode = mode.lower()
    if mode == 'euclidean':
        return np.linalg.norm(np.array(p1), np.array(p2))