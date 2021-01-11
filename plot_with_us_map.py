import joblib
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import rotate
from scipy.spatial.distance import squareform


def setup_ax(ax):
    # Spacing between each line
    intervals = 0.1

    ax.xaxis.set_major_locator(MultipleLocator(base=intervals))
    ax.yaxis.set_major_locator(MultipleLocator(base=intervals))

    # Add the grid
    ax.grid(which="both", axis="both", linestyle=":")


def plot_us_map(out_name="us_map.png"):
    map_file_name = f"{plot_dir}/usa_map_transparent.png"
    img_bg = plt.imread(map_file_name)

    ratio = 2400.0 / 1392.0
    width = 1.25
    height = width / ratio
    # width, height = 1.2, 0.6
    x_offset, y_offset = 0.05, 0.01
    img_pos = [
        -width / 2.0 + x_offset,
        width / 2.0 + x_offset,
        -height / 2.0 + y_offset,
        height / 2.0 + y_offset,
    ]  # [left, right, bottom, top]

    indicated_points = (width / 2.0) * np.array(
        [
            [-0.63, 0.46],  # Olympia
            [-0.7, 0.075],  # San Joe
            [-0.65, -0.05],  # Los Angeles
            [0.07, -0.3],  # Austin
            [-0.03, 0.37],  # Bismarch
            [0.38, 0.18],  # Chicago
            [0.715, -0.45],  # Miami
            [0.815, 0.175],  # New York City
            [0.75, 0.072],  # Washington DC
            [0.73, 0.04],  # Richmond
        ]
    ) + np.array([[x_offset, y_offset]])

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_aspect("equal")
    ax.axis("off")
    # setup_ax(ax)

    # plot background map
    ax.imshow(img_bg, extent=img_pos)

    # plot indicated points
    ax.scatter(*indicated_points.T, marker="o", s=128, c="C2")

    fig.savefig(out_name, bbox_inches="tight", transparent=True)


def plot_with_us_map_bg(Z, names, out_name="cities_aligned.png"):
    map_file_name = f"{plot_dir}/us_map.png"
    img_bg = plt.imread(map_file_name)

    ratio = 2400.0 / 1392.0
    width = 1.25
    height = width / ratio
    x_offset, y_offset = 0.035, -0.01
    img_pos = [
        -width / 2.0 + x_offset,
        width / 2.0 + x_offset,
        -height / 2.0 + y_offset,
        height / 2.0 + y_offset,
    ]  # [left, right, bottom, top]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_aspect("equal")
    # ax.axis("off")
    # setup_ax(ax)

    # plot background map
    ax.imshow(img_bg, extent=img_pos, alpha=0.3)

    # plot result embedding
    ax.scatter(*Z.T, marker="+", s=128, c="blue")
    ax.scatter(*Z[[3, 6]].T, marker="*", s=128, c="red")

    # show cities names with config for fixed point
    highlighted = dict(facecolor="none", edgecolor="green", boxstyle="round")
    for (x, y), s in zip(Z, names):
        y_offset = 0.03 if s in ["Washington DC", "Chicago", "New York"] else -0.03
        bbox_style = highlighted if s in ["Olympia", "Washington DC"] else None
        ax.text(
            x,
            y + y_offset,
            s,
            ha="center",
            va="center",
            color="blue",
            fontsize=14,
            bbox=bbox_style,
        )

    fig.savefig(out_name, bbox_inches="tight", transparent=False)


def plot_original_MDS(D, names, out_name="original.png"):
    Z = MDS(
        dissimilarity="precomputed",
        metric=True,
        random_state=2021,
        n_init=10,
        n_jobs=-1,
        verbose=1,
    ).fit_transform(squareform(D))

    map_file_name = f"{plot_dir}/us_map90.png"
    img_bg = plt.imread(map_file_name)

    ratio = 1392.0 / 2400.0
    height = 1.2
    width = height * ratio
    x_offset, y_offset = -0.075, -0.1
    img_pos = [
        -width / 2.0 + x_offset,
        width / 2.0 + x_offset,
        -height / 2.0 + y_offset,
        height / 2.0 + y_offset,
    ]  # [left, right, bottom, top]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_aspect("equal")
    # ax.axis("off")
    # setup_ax(ax)

    # plot background map
    ax.imshow(img_bg, extent=img_pos, alpha=0.3)

    # plot result embedding
    ax.scatter(*Z.T, marker="+", s=128, c="blue")

    # show cities names with config for fixed point
    for (x, y), s in zip(Z, names):
        y_offset = 0.03 if s in ["Washington DC", "Los Angeles", "New York"] else -0.03
        ax.text(x, y + y_offset, s, ha="center", va="center", color="blue", fontsize=14)

    fig.savefig(out_name, bbox_inches="tight")


if __name__ == "__main__":
    plot_dir = "./plots/cities_us_toy"

    city_names = [
        "New York",
        "San Jose",
        "Los Angeles",
        "Washington DC",
        "Miami",
        "Austin",
        "Olympia",
        "Bismarck",
        "Chicago",
        "Richmond,",
    ]

    # plot_us_map(out_name=f"{plot_dir}/us_map.png")

    Z1, dists_with_indices, labels = joblib.load("./embeddings/cities_us_toy_MAP2.Z")
    # plot_with_us_map_bg(Z1, city_names, out_name=f"{plot_dir}/cities_aligned.png")

    D, _ = list(zip(*dists_with_indices))
    plot_original_MDS(D, city_names, out_name=f"{plot_dir}/original.png")
