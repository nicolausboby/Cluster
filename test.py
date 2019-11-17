from hierarchical import agglomerative
# import dbscan
import numpy as np


def read_data(filename, return_type="list"):
    # parse csv file
    f = open(filename, 'r')

    lines = f.read().splitlines()
    matrix = []
    for line in lines:
        row = []
        tokens = line.split(",")
        for token in tokens:
            try:
                row.append(float(token))
            except ValueError:
                row.append(token)
        matrix.append(row)

    return matrix


data = np.array(read_data('iris.data'))

train = data[:, :4]
train = train.astype('float64')
# print(train)
clust = agglomerative(train.tolist(), 3, 'single', 'euclidean')
print(clust)