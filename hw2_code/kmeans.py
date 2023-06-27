'''
File: kmeans.py
Project: Downloads
File Created: Feb 2021
Author: Rohit Das
'''

import numpy as np


class KMeans(object):

    def __init__(self):  # No need to implement
        pass

    def _init_centers(self, points, K, **kwargs):  # [2 pts]
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            kwargs: any additional arguments you want
        Return:
            centers: K x D numpy array, the centers.
        Hint: Please initialize centers by randomly sampling points from the dataset in case the autograder fails.
        """
        #pick K random indices from 0 to N
        #use these indices to find random points
        return points[np.random.choice(points.shape[0], K)]

        #raise NotImplementedError

    def _kmpp_init(self, points, K, **kwargs): # [3 pts]
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            kwargs: any additional arguments you want
        Return:
            centers: K x D numpy array, the centers.
        """
        n = points.shape[0]
        #initialize centers as inf so we can find min distances 
        #w/o worrying about centers not yet computed
        centers = np.full((K,points.shape[1]),np.inf) 
        kmppPoints = points[np.random.choice(n, int(.01*n), replace=False)] #samples
        centers[0] = kmppPoints[np.random.choice(kmppPoints.shape[0], 1)]#first center
        for k in range(1,K):
            distances = pairwise_dist(kmppPoints,centers)
            min_distances = np.amin(distances,axis=1)#min distance
            centers[k] = kmppPoints[np.argmax(min_distances,axis=0)]#max min distance
        return centers
        #raise NotImplementedError

    def _update_assignment(self, centers, points):  # [10 pts]
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            points: NxD numpy array, the observations
        Return:
            cluster_idx: numpy array of length N, the cluster assignment for each point

        Hint: You could call pairwise_dist() function.
        """
        #find distance b/t points and centers
        #return N length array of indices corresponding to centers
        return np.argmin(pairwise_dist(points,centers), axis=1)
        #raise NotImplementedError

    def _update_centers(self, old_centers, cluster_idx, points):  # [10 pts]
        """
        Args:
            old_centers: old centers KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            centers: new centers, a new K x D numpy array of float dtype, where K is the number of clusters, and D is the dimension.

        HINT: Points may be integer, but the centers should not have to be. Watch out for dtype casting!
        """
        #initialize
        centers = np.zeros(old_centers.shape)
        #loop through number of clusters
        for k in range(old_centers.shape[0]):
            #find points in cluster k
            k_points = points[cluster_idx==k]
            #average the features of these points to find their new center
            #axis=0 calculates mean up and down the data
            centers[k] = np.mean(k_points, axis=0)
        return centers

        #raise NotImplementedError

    def _get_loss(self, centers, cluster_idx, points):  # [5 pts]
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            loss: a single float number, which is the objective function of KMeans.
        """
        loss = 0
        #outside summation from obj function
        for k in range(centers.shape[0]):
            #find points in cluster k
            k_points = points[cluster_idx==k]
            #calculate loss of this cluster
            #np.sum represents the inside summation from obj function
            k_loss = np.sum(np.square(pairwise_dist(k_points,centers[k])))
            #acts as summation in obj func over k centers
            loss += k_loss
        return loss
        #raise NotImplementedError

    def _get_centers_mapping(self, points, cluster_idx, centers):
        # This function has been implemented for you, no change needed.
        # create dict mapping each cluster to index to numpy array of points in the cluster
        centers_mapping = {key : [] for key in [i for i in range(centers.shape[0])]}
        for (p, i) in zip(points, cluster_idx):
            centers_mapping[i].append(p)
        for center_idx in centers_mapping:
            centers_mapping[center_idx] = np.array(centers_mapping[center_idx])
        self.centers_mapping = centers_mapping
        return centers_mapping

    def __call__(self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, verbose=False, center_mapping=False, **kwargs):
        """
        This function has been implemented for you, no change needed.

        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            verbose: boolean to set whether method should print loss (Hint: helpful for debugging)
            kwargs: any additional arguments you want
        Return:
            cluster assignments: Nx1 int numpy array
            cluster centers: K x D numpy array, the centers
            loss: final loss value of the objective function of KMeans
        """
        centers = self._init_centers(points, K, **kwargs)
        for it in range(max_iters):
            cluster_idx = self._update_assignment(centers, points)
            centers = self._update_centers(centers, cluster_idx, points)
            loss = self._get_loss(centers, cluster_idx, points)
            K = centers.shape[0]
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            if verbose:
                print('iter %d, loss: %.4f' % (it, loss))
        if center_mapping:
            return cluster_idx, centers, loss, self._get_centers_mapping(points, cluster_idx, centers)
        return cluster_idx, centers, loss


def pairwise_dist(x, y):  # [5 pts]
    np.random.seed(1)
    """
        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
                dist: N x M array, where dist2[i, j] is the euclidean distance between
                x[i, :] and y[j, :]
    """
    #x[:,np.newaxis]-y creates new axis so x-y fits broadcasting
    #square this value in accordance w/ euclidean distance formula
    #take the sum for each feature
    #sqrt in accordance w/ euclidean distance formula
    return np.sqrt(np.sum(np.square(x[:,np.newaxis]-y),axis=2))
    #raise NotImplementedError

def silhouette_coefficient(points, cluster_idx, centers, centers_mapping): # [10pts]
    """
    Args:
        points: N x D numpy array
        cluster_idx: N x 1 numpy array
        centers: K x D numpy array, the centers
        centers_mapping: dict with K keys (cluster indicies) each mapping to a C_i x D 
        numpy array with C_i corresponding to the number of points in cluster i
    Return:
        silhouette_coefficient: final coefficient value as a float 
        mu_ins: N x 1 numpy array of mu_ins (one mu_in for each data point)
        mu_outs: N x 1 numpy array of mu_outs (one mu_out for each data point)
    """


    #LOOP THROUGH ALL CLUSTERS AND FIND CLOSEST CLUSTERS THAT WAY
    
    mu_ins = np.zeros(cluster_idx.shape)
    mu_outs = np.zeros(cluster_idx.shape)
    sc = 0
    n = points.shape[0]
    s = np.zeros(n)
    k = None
    for i, point in enumerate(points):
        #mu_in
        #find what cluster we are in
        k = cluster_idx[i]
        #reshape points for pairwise distance
        point = point.reshape(1,point.shape[0])
        #follows formula for mu_in
        mu_ins[i] = (np.sum(pairwise_dist(point, centers_mapping[k])))/(centers_mapping[k].shape[0]-1)
        #mu_out
        curr_mu_out = []
        for cluster, c_points in centers_mapping.items():
            #dont want to calculate distance to own cluster
            if cluster == k:
                continue
            #find mu_outs to other cluster points
            curr_mu_out.append((np.sum(pairwise_dist(point, c_points)))/c_points.shape[0])
        #take min of the mu_out we found
        mu_outs[i] = min(curr_mu_out)
        #si
        s[i] = (mu_outs[i]-mu_ins[i])/max(mu_outs[i],mu_ins[i])
    #final sil coeff calculation
    sc = np.sum(s)/n

    return sc, mu_ins, mu_outs
