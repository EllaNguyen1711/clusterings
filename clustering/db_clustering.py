import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import pickle
import hdbscan
import hkdataminer.cluster.aplod_ as aplod
from sklearn.cluster import DBSCAN, KMeans
import pyemma.coordinates as coor


class Clustering:

	def __init__(self, FN, method, min_samples, stride, eps, min_cluster_size):
		self.FN = FN
		self.method = method
		self.min_samples = min_samples
		self.stride = stride
		self.eps = eps
		self.min_cluster_size = min_cluster_size

	def read_h5(self, dim=None):
		dt = h5.File(self.FN, 'r')
		n_dt = [np.array(dt[key][:, :dim] if dim is not None else dt[key]) for key in dt.keys()]
		index = [np.concatenate(([[i]*dt[key].shape[0]], [np.arange(dt[key].shape[0])]), axis=0).T for i, key in enumerate(dt.keys())]
		return n_dt, index

	def density_based_w_stride(self):
		read_dt = self.read_h5()
		f_data = np.concatenate(read_dt[0])[::self.stride]
		new_index = np.concatenate(read_dt[1])[::self.stride]
		if self.method == 'HDBSCAN':
			clustering = hdbscan.HDBSCAN(min_cluster_size= self.min_cluster_size, min_samples= self.min_samples,  gen_min_span_tree=True).fit(f_data)
		elif self.method == 'DBSCAN':
			if self.eps == None:
				raise ValueError('Need to provide an EPS value!!!')
			else:
				clustering = DBSCAN(eps= self.eps, min_samples= self.min_samples).fit(f_data)
		elif self.method == 'APLoD':
			clustering = aplod.APLoD(metric='euclidean', n_samples = 30000, n_neighbors = self.min_samples).fit(f_data)
		elif self.method == 'Kmeans':
			clustering = KMeans(n_clusters =self.min_samples, init='k-means++', max_iter = 50).fit(f_data)
		
		assignments = clustering.labels_
		print('Number of clusters is: ', max(assignments)+1)

		if self.method == 'HDBSCAN' or self.method == 'HDBSCAN':
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
	
			labels_tba = np.array(labels_tba)
			n_clusters = int(max(labels_tba) + 1)
			f_assignments = []
	    	
			for n in range(len(read_dt[0])):
				cl = []
				for p, order in enumerate(new_index):
					if n == order[0]:
						cl.append(labels_tba[p])
					f_assignments.append(np.array(cl))
		else:
			f_assignments = []
			for n in range(len(read_dt[0])):
				cl = []
				for p, order in enumerate(new_index):
					if n == order[0]:
						cl.append(assignments[p])
				f_assignments.append(np.array(cl))
		
		parameter = {'method': self.method, 'or_dt_len': len(np.concatenate(read_dt[0])), 'stride': self.stride, 'min_samples': self.min_samples}

		return assignments, f_assignments, parameter

	def transform(self, f_assignments):
		read_dt = self.read_h5()
		stride = self.stride
		wo_stride_assig = []
		row_indices = np.concatenate(read_dt[1])[:, 0]
		ful_len = len(np.concatenate(read_dt[0]))

		for i, vl in enumerate(np.concatenate(f_assignments)):
			if i != 0:
				wo_stride_assig.append([vl]*stride)
			elif i == 0:
				wo_stride_assig.append([vl]*((stride-1)//2))
		wo_stride_assig = np.concatenate(wo_stride_assig)
		wo_stride_assig = wo_stride_assig.tolist() + list([np.concatenate(f_assignments)[-1]]*(len(np.concatenate(read_dt[0]))-len(wo_stride_assig)))

		wo_stride_assig = np.array(wo_stride_assig)

		wo_stride_f_assignments = []
		for i in range(len(read_dt[1])):
			cl = wo_stride_assig[row_indices == i]
			wo_stride_f_assignments.append(np.array(cl))

		return wo_stride_f_assignments

