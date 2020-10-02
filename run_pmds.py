import numpy as np
from matplotlib import pyplot as plt

from sklearn.datasets import load_iris, load_digits, load_wine
from sklearn.utils import shuffle
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

from pmds import pmds


def run_pdms(D, N, labels=None):
    Z, Z_var, losses = pmds(D, n_samples=N)

    # original MDS
    Z0 = MDS(metric="precomputed").fit_transform(squareform(D))

    fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(10, 4))
    ax0.set_title("Original MDS")
    ax0.scatter(*Z0.T, c=labels, cmap="tab10")

    ax1.set_title("Probabilistic MDS")
    ax1.scatter(*Z.T, c=labels, cmap="tab10")
    fig.savefig(f"{plot_dir}/Z.png")

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.plot(losses)
    fig.savefig(f"{plot_dir}/loss.png")


if __name__ == "__main__":
    n_samples = 50
    dataset_name = "iris"
    plot_dir = f"plots/{dataset_name}"
    load_func = {"iris": load_iris, "wine": load_wine, "digits": load_digits}[
        dataset_name
    ]

    X, y = shuffle(*load_func(return_X_y=True), n_samples=n_samples)
    print(X.shape, y.shape)

    # test standardize input data
    X = StandardScaler().fit_transform(X)
    if len(X) > 10:
        X = PCA(0.95).fit_transform(X)
        print("[Dataset] After PCA: ", X.shape)

    D = pdist(X)
    res = run_pdms(D, N=len(X), labels=y)
