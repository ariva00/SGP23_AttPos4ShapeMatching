import igl
import torch
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist

# Original transmatching repository https://github.com/GiovanniTRA/transmatching
# The orignial code is distributed under the MIT license reported in the license folder

def est_area(X):
    Ds = torch.cdist(X,X)
    Ds = 1/(Ds<0.05).float().sum(-1)
    return Ds

def chamfer_loss(X,Y):
    dist = torch.cdist(X,Y)
    losses = dist.min(-1)[0].mean(-1)+dist.min(-2)[0].mean(-1)
    return losses

def get_errors(d, gt_mat):

    p2p = torch.argmin(d, dim=-1).cpu()

    err = np.empty(p2p.shape[0])
    for i in range(p2p.shape[0]):
        pred = p2p[i]
        err[i] = gt_mat[pred, i]

    return err

def approximate_geodesic_distances(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Compute the geodesic distances approximated by the dijkstra method weighted by
    euclidean edge length
    Args:
        v: the mesh points
        f: the mesh faces
    Returns:
        an nxn matrix which contains the approximated distances
    """

    a = igl.adjacency_matrix(f)
    dist = cdist(v, v)
    values = dist[np.nonzero(a)]
    matrix = sparse.coo_matrix((values, np.nonzero(a)), shape=(v.shape[0], v.shape[0]))
    d = dijkstra(matrix, directed=False)
    return d
