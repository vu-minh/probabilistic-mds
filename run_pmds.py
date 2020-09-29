import numpy as np
from matplotlib import pyplot as plt
from numpy.lib.npyio import load
from sklearn.datasets import load_iris, load_digits, load_wine
from sklearn.utils import shuffle
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

from scipy.spatial.distance import pdist, squareform
from pmds import pmds


def run_pdms(D, N, labels=None):
    Z, Z_var = pmds(D, n_samples=N)

    # original MDS
    Z0 = MDS(metric="precomputed").fit_transform(squareform(D))

    fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(10, 4))
    ax0.set_title("Original MDS")
    ax0.scatter(*Z0.T, c=labels, cmap="tab10")

    ax1.set_title("Probabilistic MDS")
    ax1.scatter(*Z.T, c=labels, cmap="tab10")
    fig.savefig("plots/test0.png")


if __name__ == "__main__":
    X, y = shuffle(*load_wine(return_X_y=True), n_samples=100)
    print(X.shape, y.shape)

    # test standardize input data
    X = StandardScaler().fit_transform(X)

    D = pdist(X)
    res = run_pdms(D, N=len(X), labels=y)
