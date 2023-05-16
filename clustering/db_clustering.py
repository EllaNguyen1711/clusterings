import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import pickle
import hdbscan
import hkdataminer.cluster.aplod_ as aplod
from sklearn.cluster import DBSCAN, KMeans
import pyemma.coordinates as coor


def TBA(assignments):
    labels_tba = []
    for i, l in enumerate(assignments):
        if l == -1:
            if len(labels_tba) > 0:
                labels_tba.append(labels_tba[-1])
            else:
                for k, vl in enumerate(assignments):
                    if vl > -1:
                        labels_tba.append(vl)
                        break
        else:
            labels_tba.append(l)
    return np.array(labels_tba)


class READ_h5:

    def __init__(self, FN):
        self.FN = FN

    def read_h5(self, dim=None):
        dt = h5.File(self.FN, 'r')
        n_dt = [np.array(dt[key][:, :dim] if dim is not None else dt[key])
                for key in dt.keys()]
        index = [np.concatenate(([[i]*dt[key].shape[0]], [np.arange(dt[key].shape[0])]),
                                axis=0).T for i, key in enumerate(dt.keys())]
        return n_dt, index


class Clustering:

    def __init__(self, data, index, method, sel_k, stride):
        self.data = data
        self.index = index
        self.method = method
        self.sel_k = sel_k
        self.stride = stride

    def density_based_w_stride(self, max_iter=300, n_samples=1000, eps=0.3):
        f_data = np.concatenate(self.data)[::self.stride]
        new_index = np.concatenate(self.index)[::self.stride]
        if self.method == 'HDBSCAN':
            clustering = hdbscan.HDBSCAN(
                min_cluster_size=self.sel_k,  gen_min_span_tree=True).fit(f_data)
        elif self.method == 'DBSCAN':
            if eps == None:
                raise ValueError('Need to provide an EPS value!!!')
            else:
                clustering = DBSCAN(
                    eps=eps, min_samples=self.sel_k).fit(f_data)
        elif self.method == 'APLoD':
            clustering = aplod.APLoD(
                metric='euclidean', n_samples=n_samples, n_neighbors=self.sel_k).fit(f_data)
        elif self.method == 'Kmeans':
            clustering = None
            i = 0
            while i < 3:
                kmeans = KMeans(n_clusters=self.sel_k, max_iter=max_iter)
                clustering = kmeans.fit(f_data)
                print('Number of iterations have been done: ', clustering.n_iter_)
                if clustering.n_iter_ < kmeans.max_iter:
                    break
                i += 1
                print('Repeats of Kmeans: ', i)

        assignments = np.array(clustering.labels_)
        return assignments

    def get_dtrajs(self, max_iter= 300, n_samples = 1000, eps=0.3):
        new_index = np.concatenate(self.index)[::self.stride][:, 0]
        raw_assignments = self.density_based_w_stride(max_iter=max_iter, n_samples=n_samples, eps=eps)

        if self.method == 'HDBSCAN' or self.method == 'DBSCAN':
            labels_tba = TBA(raw_assignments)
            f_assignments = []
            for n in range(len(self.index)):
                cl = labels_tba[new_index == n]
                f_assignments.append(np.array(cl))
        else:
            f_assignments = []
            for n in range(len(self.index)):
                cl = raw_assignments[new_index == n]
                f_assignments.append(np.array(cl))

        return f_assignments

    def transform_to_full(self):
        f_assignments = self.get_dtrajs()
        fin_assignments = []
        for j, c in enumerate(self.index):
            if len(c) <= len(f_assignments[j])*self.stride:
                wo_stride_cl = np.concatenate(
                    [[n]*self.stride for n in f_assignments[j]])[:len(c)]
            else:
                t1 = np.concatenate(
                    [[n]*self.stride for n in f_assignments[j]])
                t2 = np.concatenate(
                    [[f_assignments[j][-1]]*(len(c) - len(f_assignments[j])*self.stride)])
                wo_stride_cl = np.concatenate([t1, t2])
            fin_assignments.append(wo_stride_cl)
        return fin_assignments


if __name__ == "__main__":  # pragma: no cover
    # Do something if this file is invoked on its own
    import time
    import os
    import numpy as np

    mod_path = os.getcwd()
    FN = mod_path + '/data/test.h5'

    read_dt = READ_h5(FN)
    data = read_dt.read_h5()[0]
    index = read_dt.read_h5()[1]
    print (index)
    method = 'HDBSCAN'
    sel_k = 5
    stride = 2

    start_time = time.time()

    clustering = Clustering(data, index, method, sel_k, stride)
    dtrajs = clustering.transform_to_full()
    np.save(mod_path + '/data/dtrajs_test.npy', dtrajs)
    print("--- %s seconds ---" % (time.time() - start_time))
