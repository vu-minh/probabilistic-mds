from sklearn.manifold._mds import _smacof_single


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
