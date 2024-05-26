
import ctypes

import numpy as np
import scipy.cluster.hierarchy as hcluster
import scipy.spatial as spatial

from numpy.ctypeslib import ndpointer
from sklearn.mixture import GaussianMixture


def cluster_points(points, cluster_distance=1.0):
    tree = spatial.KDTree(points)

    search_inds = np.arange(len(points))

    clusters = []
    cluster_inds = []

    while len(search_inds) != 0:
        cluster = tree.query_ball_point(points[search_inds[0], :],
                                        cluster_distance)
        clusters.append(np.mean(points[cluster, :], axis=0))
        cluster_inds.append(np.array(cluster))
        search_inds = np.setdiff1d(search_inds, cluster)

    clusters = np.array(clusters)
    cluster_inds = np.array(cluster_inds)

    return clusters, cluster_inds


def gaussian_mixture(points, num_clusters=2):
    gm = GaussianMixture(n_components=num_clusters, random_state=0).fit(points)
    return gm


def cluster_hierarchical(points, cluster_distance=1.0):
    clusters_indices = hcluster.fclusterdata(points, cluster_distance,
                                             criterion='distance')
    cluster_vals = np.unique(clusters_indices)
    clusters = []
    cluster_inds = []
    for cval in cluster_vals:
        cinds = np.where(clusters_indices == cval)[0]
        clusters.append(points[cinds, :].mean(0))
        cluster_inds.append(cinds)
    clusters = np.array(clusters)
    return clusters, cluster_inds


def in_rectangles(points, rectangles):
    intersections = []
    for point in points:
        intersections2 = []
        for rect in rectangles:
            area = (rect[2] - rect[0]) * (rect[3] - rect[1])
            x_in = rect[0] < point[0] < rect[2]
            y_in = rect[1] < point[1] < rect[3]
            if x_in and y_in:
                intersections2.append([True, area])
            else:
                intersections2.append([False, area])
        intersections.append(intersections2)
    return intersections


def num_neighbors(points, distance, dll_path):
    nd = ctypes.CDLL(dll_path)
    num_neigh = nd.numberOfNeighbors
    num_neigh.restype = int
    num_neigh.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ctypes.c_int32, ctypes.c_float,
                              ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS")]

    neighbor_count = np.zeros(len(points), dtype='int')

    success = num_neigh(points.astype('float32'), len(points), float(distance),
                        neighbor_count)

    return neighbor_count


def fit_plane(points, num_iters=3):
    """
    Fits a plane to an array of XY coordinates
    :param points: The points to fit the plane to
    :param num_iters: The number of iterations of outlier removal
    :return:
    """
    from skspatial.objects import Plane
    plane = Plane.best_fit(points)
    for ix in range(num_iters - 1):
        dists = np.array([plane.distance_point(p) for p in points])
        d_mean = dists.mean()
        d_std = dists.std()
        inds = np.where(dists < (d_mean + d_std))[0]
        plane = Plane.best_fit(points[inds])
    return plane.cartesian(), plane
