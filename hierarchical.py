"""
Hierarchical clustering 

"""
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage

def agglomerative():
    pass

def _linkage(c1, c2, mode = 'single'):
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
        valid modes : single, complete, avg, avg_group

    Returns
    -------
    d : float
        Dissimilarity result between 2 clusters, calculated based on linkage mode
    """
    if mode ==  'single':
        for p in c1:
            pass

def _distance(p1, p2, mode='euclidean'):
    pass