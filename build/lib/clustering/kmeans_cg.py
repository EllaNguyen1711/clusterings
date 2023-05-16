import h5py as h5
import numpy as np
from scipy.spatial import cKDTree
from itertools import islice
from multiprocessing import Pool
import pickle
import pyemma.coordinates as coor


def kmeans(data, n_clusters):
    data = list(data)
    cl_ = coor.cluster_kmeans(data, k=n_clusters, max_iter=50)
    fin_dtrajs = cl_.dtrajs
    cluster_centers = cl_.cluster_centers_
    cluster_ind = cl_.index_clusters
    return fin_dtrajs, cluster_centers, cluster_ind


def form_clindexes(dtrajs):
    # Convert dtrajs of assigments to cluster indexes
    cl = [[] for _ in range(max(np.concatenate(dtrajs))+1)]
    for j, traj in enumerate(dtrajs):
        for ind, i in enumerate(traj):
            cl[i].append([j, ind])
    return cl


def find_k_closest(centroids, data, k=1, distance_norm=2):
    kdtree = cKDTree(data, leafsize=16)
    distances, indices = kdtree.query(centroids, k, p=distance_norm)
    if k > 1:
        indices = indices[:, -1]
    values = data[indices]
    return indices, values


def cluster_tica(dt_FN, cl_index, center):
    # Rewrite tica data assigned to cluster #n
    with h5.File(dt_FN, 'r') as tica_data:
        cluster_tica = np.array(
            [tica_data['%04d' % ind[0]][ind[1]] for ind in cl_index])
    indices, values = find_k_closest(center, cluster_tica)
    fin_ind = cl_index[indices]
    return fin_ind, indices, values


class Transform:
    # Transform tica data from trajs to clusters in order to be ready for the next clustering

    def __init__(self, dt_FN, cluster_ind, cluster_centers):
        self.dt_FN = dt_FN
        self.cluster_ind = cluster_ind
        self.cluster_centers = cluster_centers
        self.var = [[self.dt_FN, self.cluster_ind[i], ct]
                    for i, ct in enumerate(self.cluster_centers)]

    def chunked_iterable(self, iterable, size):
        it = iter(iterable)
        while True:
            chunk = list(islice(it, size))
            if not chunk:
                return
            yield chunk

    def cluster_data_multiprocessing(self, size=10):
        ind = []
        tic = []
        with Pool() as p:
            traj_list = p.map(cluster_tica, self.var, chunksize=size)
            ind.append([val[0] for val in traj_list])
            tic.append([val[2] for val in traj_list])
        return ind, tic
