import numpy as np
from matplotlib import pyplot as plt
from numpy.lib.npyio import load
from sklearn.datasets import load_iris
from sklearn.utils import shuffle

from scipy.spatial.distance import pdist, squareform
from pmds import pmds


def run_pdms(D, N, labels=None):
    Z, Z_var = pmds(D, n_samples=N)

    plt.figure(figsize=(4, 4))
    plt.scatter(*Z.T, c=labels)
    plt.savefig("plots/test0.png")


if __name__ == "__main__":
    X, y = shuffle(*load_iris(return_X_y=True), n_samples=25)
    print(X.shape, y.shape)

    D = squareform(pdist(X))
    print(D.shape)

    res = run_pdms(D, N=len(X), labels=y)