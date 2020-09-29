import numpy as np
from matplotlib import pyplot as plt
from numpy.lib.npyio import load
from sklearn.datasets import load_iris, load_digits, load_wine
from sklearn.utils import shuffle
from sklearn.manifold import MDS

from scipy.spatial.distance import pdist, squareform
from pmds import pmds


def run_pdms(D, N, labels=None):
    Z, Z_var = pmds(D, n_samples=N)

    # original MDS
    Z0 = MDS(metric="precomputed").fit_transform(D)

    fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(10, 4))
    ax0.set_title("Original MDS")
    ax0.scatter(*Z0.T, c=labels, cmap="tab10")

    ax1.set_title("Probabilistic MDS")
    ax1.scatter(*Z.T, c=labels, cmap="tab10")
    fig.savefig("plots/test0.png")


if __name__ == "__main__":
    X, y = shuffle(*load_iris(return_X_y=True), n_samples=25)
    print(X.shape, y.shape)

    D = squareform(pdist(X))
    print(D.shape)

    res = run_pdms(D, N=len(X), labels=y)
