from sklearn.manifold._mds import _smacof_single
from scipy.spatial.distance import pdist, squareform


EPS = 1e-8


def stress(D, Z):
    """Metric MDS stress score.
    See: https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/manifold/_mds.py#L110

    Parameters
    ----------
    D : ndarray[float] (n_samples, n_samples)
        Squareform pairwise distances in HD space.
    Z : ndarray[float] (n_samples, n_components)
        Embedding in LD space.
    """
    _, stress, _ = _smacof_single(
        dissimilarities=D, metric=True, n_components=2, init=Z, max_iter=1
    )
    return stress


def sammonD(D, d):
    return ((D - d) ** 2 / (D + EPS)).sum() / (D.sum() + EPS)


def sammonZ(D, Z):
    if len(D.shape) == 2:
        D = squareform(D)
    return sammonD(D, pdist(Z))


def all_scores(D, Z, to_string=True):
    res = {"mds": stress(D, Z), "sammon": sammonZ(D, Z)}
