import numpy as np
from kmeans import pairwise_dist

class DBSCAN(object):
    
    def __init__(self, eps, minPts, dataset):
        self.eps = eps
        self.minPts = minPts
        self.dataset = dataset
        
    def fit(self):
        """Fits DBSCAN to dataset and hyperparameters defined in init().
        Args:
            None
        Return:
            cluster_idx: (N, ) int numpy array of assignment of clusters for each point in dataset
        Hint: Using sets for visitedIndices may be helpful here.
        Iterate through the dataset sequentially and keep track of your points' cluster assignments.
        If a point is unvisited or is a noise point (has fewer than the minimum number of neighbor points), then its cluster assignment should be -1.
        Set the first cluster as C = 0
        """
        #initalize to -1 since we add to c before calling expand cluster
        c = -1
        visitedIndices = set()
        #initialize as -1 since we want unvisited points to have c=-1
        cluster_idx = np.full(self.dataset.shape[0],-1)
        for i, point in enumerate(self.dataset):
            if i not in visitedIndices:
                #we visit this point
                visitedIndices.add(i)
                #find neighbors of the point
                neighborIndices = self.regionQuery(i)
                #if not noise, expand cluster
                if len(neighborIndices) >= self.minPts:
                    c += 1
                    self.expandCluster(i, neighborIndices, c, cluster_idx, visitedIndices)
        return cluster_idx
        #raise NotImplementedError

    def expandCluster(self, index, neighborIndices, C, cluster_idx, visitedIndices):
        """Expands cluster C using the point P, its neighbors, and any points density-reachable to P and updates indices visited, cluster assignments accordingly
           HINT: regionQuery could be used in your implementation
        Args:
            index: index of point P in dataset (self.dataset)
            neighborIndices: (N, ) int numpy array, indices of all points witin P's eps-neighborhood
            C: current cluster as an int
            cluster_idx: (N, ) int numpy array of current assignment of clusters for each point in dataset
            visitedIndices: set of indices in dataset visited so far
        Return:
            None
        Hints: 
            np.concatenate(), np.unique(), np.sort(), and np.take() may be helpful here
            A while loop may be better than a for loop
        """
        cluster_idx[index] = C
        #keep adding and removing neighbors until none left
        while len(neighborIndices) > 0:
            #take neighbor at end of array (helps with while loop)
            nIndex = neighborIndices[-1]
            #remove this point from neighborIndices
            neighborIndices = neighborIndices[:-1]
            if nIndex not in visitedIndices:
                visitedIndices.add(nIndex)
                nPoints = self.regionQuery(nIndex)
                #make sure current point does not get added back into neighbors
                nPoints = np.delete(nPoints, np.where(nPoints==nIndex))
                if len(nPoints) >= self.minPts:
                    neighborIndices = np.unique(np.concatenate((neighborIndices,nPoints)))
            #c=-1 if unvisited or noise (not in cluster yet)
            if cluster_idx[nIndex] == -1:
                cluster_idx[nIndex] = C
        #raise NotImplementedError
        
    def regionQuery(self, pointIndex):
        """Returns all points within P's eps-neighborhood (including P)

        Args:
            pointIndex: index of point P in dataset (self.dataset)
        Return:
            indices: (I, ) int numpy array containing the indices of all points within P's eps-neighborhood
        Hint: pairwise_dist (implemented above) and np.argwhere may be helpful here
        """
        data = self.dataset
        point = data[pointIndex]
        point = point.reshape(1,point.shape[0])
        distances = pairwise_dist(point, data)
        #finds indices of points within eps
        indices = np.argwhere(distances<=self.eps)[:,1]
        return indices
        #raise NotImplementedError