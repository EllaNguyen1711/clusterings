import numpy as np
import h5py as h5
import pyemma.coordinates as coor
import matplotlib.pyplot as plt
from wstride_clustering import Clustering


def test_Clustering():
    test_FN = 'data/chi1_2_tica_10_dim.h5'
    stride = 100
    # Initialize the clustering object
    cl = Clustering(test_FN, method='DBSCAN', min_samples=100, stride=stride, eps= 0.45, min_cluster_size = None)

    # Test the read_h5 method
    n_dt, index = cl.read_h5(dim=5)
    assert n_dt[0][0].shape == (5,)
    assert index[0][0].shape == (2,)

    # Test the density_based_w_stride method
    assignments, f_assignments, parameter = cl.density_based_w_stride()
    assert len(assignments) == len(np.concatenate(n_dt)[::stride])
    assert len(f_assignments) == len(n_dt)
    assert isinstance(parameter, dict)

    # Test the transform method
    wo_stride_f_assignments = cl.transform(f_assignments)
    assert len(wo_stride_f_assignments) == len(n_dt)
    assert len(wo_stride_f_assignments[0]) == len(n_dt[0])
    assert wo_stride_f_assignments[0][0].shape == (5,)

    # Plot the results
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i in range(len(n_dt)):
        ax = axs.flatten()[i]
        ax.scatter(np.concatenate(f_assignments), np.concatenate(n_dt)[::stride, i], c=assignments)
        ax.set_xlabel('Original frame index')
        ax.set_ylabel(f'Feature {i+1}')
        ax.set_title('DBSCAN clustering')

    plt.show()


if __name__ == "__main__":
    test_Clustering()