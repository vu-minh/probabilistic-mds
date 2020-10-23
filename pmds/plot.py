import mlflow
import numpy as np
import matplotlib.pyplot as plt


def line(points, out_name="line.png"):
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.plot(points)
    fig.savefig(out_name, bbox_inches="tight")
    mlflow.log_artifact(out_name)


def scatter(Z, Z_vars=None, labels=None, title="", ax=None, out_name="Z.png"):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6)) if ax is None else (None, ax)
    # ax.set_xticks([])
    # ax.set_yticks([])

    p = {} if Z_vars is None else {"marker": "+"}
    if labels is None:
        ...
    elif isinstance(labels[0], str):
        for (x, y), s in zip(Z, labels):
            ax.text(x, y, s, ha="center", va="center")
    else:
        p.update({"marker": "o", "c": labels, "cmap": "tab10"})

    ax.set_title(title)
    ax.scatter(*Z.T, **p)

    if Z_vars is not None and Z_vars.shape == (Z.shape[0],):
        p.update({"marker": "o"})
        ax.scatter(*Z.T, s=Z_vars * 500, alpha=0.05, **p)

    if fig is not None:
        fig.savefig(out_name, bbox_inches="tight")


def compare_scatter(Z0, Z1, Z0_vars, Z1_vars, labels, titles, out_name="compare.png"):
    """compare_scatter.

    Parameters
    ----------
    Z0, Z1 : ndarray, shape (n_samples, n_components)
        First and second embeddings to compare
    Z0_vars, Z1_vars : ndarray, shape (n_samples,) or (n_samples, n_components)
        Variances of Z0 and Z1
    labels : array-like (n_samples, ) or None
        Labels of points, can be int, float or str
    titles : list or None
        Title for 2 embedding
    out_name : str
        path to output figure
    """
    titles = titles or ["MDS0", "MDS1"]
    fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(10, 4))
    scatter(Z0, Z0_vars, labels, titles[0], ax=ax0)
    scatter(Z1, Z1_vars, labels, titles[1], ax=ax1)

    fig.savefig(out_name, bbox_inches="tight")
    mlflow.log_artifact(out_name)
