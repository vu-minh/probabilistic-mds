import mlflow
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


def line(points, out_name="line.png"):
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.plot(points)
    fig.savefig(out_name, bbox_inches="tight")
    mlflow.log_artifact(out_name)


def plot_losses(all_losses, titles=[], out_name="loss.png"):
    """
    Plot losses: total loss, log likelihood and log prior
    """
    fig, axes = plt.subplots(3, 1, figsize=(6, 6))
    colors = [f"C{i+1}" for i in range(len(all_losses))]
    for ax, loss, title, color in zip(axes, all_losses, titles, colors):
        ax.plot(loss, c=color)
        ax.set_title(title)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_name)


def plot_debug_Z(all_Z, labels=None, out_name="Zdebug.png", magnitude_factor=1.0):
    """Debug the evolution of the embedding in `all_Z`: [{'epoch': epoch, 'Z': Z}]
    Plot the direction of gradient from `Z_{i}` -> `Z_{i+1}`
    """
    n_plots = len(all_Z) - 1
    n_cols = 5
    n_rows = int(n_plots / n_cols + 0.5)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    for i, (Z0, Z1) in enumerate(zip(all_Z[:-1], all_Z[1:])):
        ax = axes.ravel()[i]

        ax.set_title(f"Epoch {Z0['epoch']} --> {Z1['epoch']}")
        ax.scatter(*Z0["Z"].T, c=labels, alpha=0.5)

        # plot arrow for the direction of movement
        arrowprops = dict(arrowstyle="->", alpha=0.1)
        for [x0, y0], [x1, y1] in zip(Z0["Z"], Z1["Z"]):
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=arrowprops)

    fig.tight_layout()
    fig.savefig(out_name)


def scatter(Z, Z_std=None, labels=None, title="", ax=None, out_name="Z.png"):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6)) if ax is None else (None, ax)
    # ax.set_aspect("equal")
    # ax.set_xticks([])
    # ax.set_yticks([])
    cmap = "tab10" if (labels is not None and len(np.unique(labels)) > 5) else "jet"
    marker_default_size = rcParams["lines.markersize"]

    p = {} if Z_std is None else {"marker": "+"}
    if labels is None:
        ...
    elif isinstance(labels[0], str):
        for (x, y), s in zip(Z, labels):
            ax.text(x, y, s, ha="center", va="center")
    else:
        p.update({"marker": "o", "c": labels, "cmap": cmap})

    ax.set_title(title)
    ax.scatter(*Z.T, **p)

    if Z_std is not None and Z_std.shape == (Z.shape[0],):
        # determine size for uncertainty around each point
        std_size = marker_default_size * (1.0 + Z_std * 10)
        print(
            "Z_std",
            np.min(Z_std),
            np.max(Z_std),
            "DEFAULT size: ",
            marker_default_size,
            "std_size: ",
            np.max(std_size),
            np.min(std_size),
        )
        p.update({"marker": "o", "s": std_size ** 2})
        ax.scatter(*Z.T, alpha=0.08, **p)

    if fig is not None:
        fig.savefig(out_name, bbox_inches="tight", transparent=True)


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
