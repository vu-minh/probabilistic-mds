import mlflow
import numpy as np
import matplotlib.pyplot as plt


def line(points, out_name="line.png"):
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.plot(points)
    fig.savefig(out_name, bbox_inches="tight")
    mlflow.log_artifact(out_name)


def scatter(Z, labels, title="", ax=None, out_name="Z.png"):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6)) if ax is None else (None, ax)

    p = {}
    if labels is None:
        ...
    elif isinstance(labels[0], str):
        for (x, y), s in zip(Z, labels):
            ax.text(x * 0.85, y, s)
    else:
        p = dict(c=labels, cmap="tab10")

    ax.set_title(title)
    ax.scatter(*Z.T, **p)
    if fig is not None:
        fig.savefig(out_name, bbox_inches="tight")


def compare_scatter(Z0, Z1, labels, titles, out_name="compare.png"):
    fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(10, 4))
    scatter(Z0, labels, titles[0], ax=ax0)
    scatter(Z1, labels, titles[1], ax=ax1)

    fig.savefig(out_name, bbox_inches="tight")
    mlflow.log_artifact(out_name)
