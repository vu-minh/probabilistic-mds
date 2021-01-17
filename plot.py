import math
import base64
import joblib
from io import BytesIO

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import to_rgb

import plotly.express as px
import plotly.graph_objects as go


def line(points, out_name="line.png"):
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.plot(points)
    fig.savefig(out_name, bbox_inches="tight")


def plot_hist(D, out_name="hist.png"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(D, bins=100)
    fig.savefig(out_name)


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
    ax.set_aspect("equal")
    # ax.set_xticks([])
    # ax.set_yticks([])
    cmap = "tab10" if (labels is not None and len(np.unique(labels)) <= 10) else "jet"
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
        p.update({"marker": "o", "s": std_size ** 2})
        ax.scatter(*Z.T, alpha=0.08, **p)

    if fig is not None:
        fig.savefig(out_name, bbox_inches="tight", transparent=True)


def scatter_plotly(Z, labels, out_name="Z.html"):
    fig = px.scatter(
        x=Z[:, 0],
        y=Z[:, 1],
        color=labels,
        hover_name=np.arange(len(Z)),
        color_continuous_scale="jet",
        template="simple_white",
        width=700,
        height=600,
    )
    fig.write_html(out_name)


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


SVG_META_DATA = """<?xml version="1.0" encoding="utf-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<!-- Created with matplotlib (http://matplotlib.org/),
    modified to stack multiple svg elemements,
    used for packing all images in a dataset.
    Author: Minh Vu, 2017 - 2020
-->
<svg version="1.1" width="28" height="28" viewBox="0 0 28 28"
    xmlns="http://www.w3.org/2000/svg"
    xmlns:xlink="http://www.w3.org/1999/xlink">
<defs>
<style type="text/css">
    *{stroke-linecap:butt;stroke-linejoin:round;}
    .sprite { display: none;}
    .sprite:target { display: block; margin-left: auto; margin-right: auto; }
    .packed-svg-custom {/*override this css to customize style for svg image*/}
</style>
</defs>
"""

SVG_IMG_TAG = """
<g class="sprite" id="{}">
    <image class="packed-svg-custom"
        id="stacked_svg_img_{}"
        width="28"
        height="28"
        xlink:href="data:image/png;base64,{}"
    />
</g>
"""


def generate_stacked_svg(svg_out_name, dataset, labels=None, default_cmap="gray_r"):
    """Create an SVG to store all image of a `dataset`.
    To access an image, use svg_out_name.svg#img_id, e.g. MNIST.svg#123
    """
    # current_dpi = plt.gcf().get_dpi()
    # fig = plt.figure(figsize=(28 / current_dpi, 28 / current_dpi))

    # import seaborn as sns

    # def colorize(d, color, alpha=1.0):
    #     # Using: `colorize(d.reshape(size,size), colors[t], 0.9)`
    #     rgb = np.dstack((d, d, d)) * color
    #     return np.dstack((rgb, d * alpha)).astype(np.uint8)

    def _create_cm(basecolor):
        colors = [(1, 1, 1), to_rgb(basecolor), to_rgb(basecolor)]  # R->G->B
        return LinearSegmentedColormap.from_list(colors=colors, name=basecolor)

    def _create_custom_cmap():
        basecolors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        return list(map(_create_cm, basecolors))

    def _generate_figure_data(img, cmap):
        fig_file = BytesIO()
        # plt.imshow(img, cmap=cmap)
        # plt.savefig(fig_file, transparent=True)
        plt.imsave(fig_file, img, cmap=cmap)
        plt.gcf().clear()
        fig_file.seek(0)
        return base64.b64encode(fig_file.getvalue()).decode("utf-8")

    # def _generate_figure_data2(img, color):
    #     fig_file = BytesIO()
    #     plt.imsave(fig_file, colorize(img, color, 0.9))
    #     plt.gcf().clear()
    #     fig_file.seek(0)
    #     return base64.b64encode(fig_file.getvalue()).decode("utf-8")

    N, D = dataset.shape
    img_size = int(math.sqrt(D))
    # colors = sns.color_palette("tab10")
    custom_cmap = _create_custom_cmap()

    with open(svg_out_name, "w") as svg_file:
        svg_file.write(SVG_META_DATA)

        for i in range(N):
            img = dataset[i].reshape(img_size, img_size)
            cmap = default_cmap if labels is None else custom_cmap[int(labels[i]) % 10]
            fig_data = _generate_figure_data(img, cmap)
            # color = "black" if labels is None else colors[labels[i]]
            # fig_data = _generate_figure_data2(img, color=color)
            svg_file.write(SVG_IMG_TAG.format(i, i, fig_data))

        svg_file.write("</svg>")


def stylize_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.xaxis.set_tick_params(top="off", direction="out", width=1)
    ax.yaxis.set_tick_params(right="off", direction="out", width=1)


def plot_score_with_missing_pairs(score_file_name, out_name="score.png"):
    df = pd.read_csv(score_file_name)
    print(df)
    df_grouped = df.groupby(["missing_percent"], as_index=True)
    df_summary = df_grouped.agg({"stress": ["mean", "std"]})
    df_summary.columns = ["_".join(col) for col in df_summary.columns.values]
    df_summary = df_summary.reset_index()
    print(df_summary)

    fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))
    # stylize_axes(ax)

    df_summary.plot(
        x="missing_percent",
        y="stress_mean",
        yerr="stress_std",
        ax=ax,
        legend=False,
        marker="o",
        markersize=2,
        # color="C1",
        capsize=3,
        capthick=1,
        ecolor="orange",
    )

    ax.tick_params(axis="y", direction="out", pad=-37)
    ax.set_xlabel("Percent of missing pairs")
    ax.set_ylabel("Metric MDS stress")
    fig.savefig(out_name, bbox_inches="tight")


def plot_Z_with_missing_pairs(
    embedding_dir, missing_percents, labels, out_name="all_Z.png"
):
    ncols = len(missing_percents) + 1
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(3.5 * ncols, 3))
    in_names = ["original"] + missing_percents

    for i, (ax, percent) in enumerate(zip(axes.ravel(), in_names)):
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))

        ax_idx = chr(ord("a") + i)
        if percent == "original":
            ax.set_xlabel(f"({ax_idx}) Original non-probabilistic MDS")
        else:
            ax.set_xlabel(f"({ax_idx}) PMDS with p={percent}%")

        Z, stress = joblib.load(f"{embedding_dir}/{percent}.z")
        ax.set_title(f"Stress = {stress:.2f}")
        ax.scatter(*Z.T, c=labels, alpha=0.5, cmap="tab10")

    fig.savefig(out_name, bbox_inches="tight")


def plot_scatter_with_fixed_points(
    dataset_name, Z0, Z1, fixed_points, labels, stresses, out_name
):
    if dataset_name == "automobile":
        plot_automobile_dataset(Z0, Z1, fixed_points, labels, stresses, out_name)
    elif dataset_name.startswith(("digits", "fmnist")):
        plot_image_dataset(
            dataset_name, Z0, Z1, fixed_points, labels, stresses, out_name
        )


def plot_automobile_dataset(
    Z0, Z1, fixed_points, labels, stresses, out_name="automobile.png"
):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    # TODO make grid 4x4, 4x4, 1x3 legend and 3x3 axes meaning
    Z = np.concatenate((Z0, Z1), axis=0)
    xlims = 1.1 * np.percentile(Z[:, 0], [0, 100])
    ylims = 1.1 * np.percentile(Z[:, 1], [0, 100])

    marker_styles = [
        dict(
            marker="^", color="#D0D0D0", edgecolor="#4863A0", zorder=10
        ),  # 0: 4-door, many cyl
        dict(
            marker="^", color="white", edgecolor="#4863A0", zorder=9
        ),  # 1: 2-door, many cyl
        dict(marker="o", color="#D0D0D0", edgecolor="#2F4F4F"),  # 2: 4-door, few cyl
        dict(marker="o", color="white", edgecolor="#2F4F4F"),  # 3: 2-door, few cyl
    ]

    def _scatter(ax, Z):
        for lbl in np.unique(labels):
            ax.scatter(*Z[labels == lbl].T, s=128, **marker_styles[lbl])

    for i, [ax, Z, stress] in enumerate(zip(axes.ravel(), [Z0, Z1], stresses)):
        _style_axes(ax, show_coordinates=True, hide_ticks=False)
        _scatter(ax, Z)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_title(f"Stress: {stress:.2f}")

    fixed_indices, des_pos = list(zip(*fixed_points))
    src_pos = Z0[fixed_indices, :]
    des_pos = np.array(des_pos)
    _show_moved_points(axes[0], src_pos, des_pos)

    style_fixed_points = [marker_styles[labels[i]] for i in fixed_indices]
    _show_fixed_points(axes[1], np.array(Z1)[list(fixed_indices)], style_fixed_points)

    fig.savefig(out_name, bbox_inches="tight")


def _style_axes(ax, show_coordinates=True, hide_ticks=False):
    if show_coordinates:
        ax.axhline(y=0, color="#A9A9A9", linestyle="--", alpha=0.7, zorder=99)
        ax.axvline(x=0, color="#A9A9A9", linestyle="--", alpha=0.7, zorder=99)
    if hide_ticks:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)


def _show_moved_points(ax, src_pos, des_pos, annote_src=True, annote_des=True):
    # show arrow from src to des
    arrowprops = dict(arrowstyle="<-", linestyle="--")
    for src, des in zip(src_pos, des_pos):
        ax.annotate(text="", xy=src, xytext=des, arrowprops=arrowprops, zorder=998)

    # show source points
    if annote_src:
        ax.scatter(*src_pos.T, marker="+", s=64, c="#800080", linewidths=3, zorder=999)

    if annote_des:
        # show destination points
        style = dict(facecolor="white", linestyle="--", alpha=0.35)
        ax.scatter(*des_pos.T, marker="o", s=48, color="#800080", zorder=999, **style)


def _show_fixed_points(ax, points, styles):
    for [x, y], style in zip(points, styles):
        style.update(dict(edgecolor="#800080", zorder=999))
        ax.scatter(x, y, s=128, **style)
        ax.scatter(x, y, s=64, color="#800080", marker="+", linewidths=3, zorder=1000)


def plot_image_dataset(
    dataset_name, Z0, Z1, fixed_points, labels, stresses, out_name="Z.png"
):
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 11)
    ax0 = fig.add_subplot(gs[0, 0:4])
    ax1 = fig.add_subplot(gs[0, 4:8])
    ax2 = fig.add_subplot(gs[0, 8:])

    Z = np.concatenate((Z0, Z1), axis=0)
    xlims = 1.05 * np.percentile(Z[:, 0], [0, 100])
    ylims = 1.05 * np.percentile(Z[:, 1], [0, 100])

    X = joblib.load(f"./embeddings/{dataset_name}_data.Z")
    img_size = int(math.sqrt(X.shape[1]))
    X = X.reshape(-1, img_size, img_size)

    def _scatter_image(ax, Z, zoom=0.375, cmap="binary", fixed_indices=[]):
        def _add_artist(img, pos, **style):
            ab = AnnotationBbox(OffsetImage(img, zoom=zoom, cmap=cmap), pos, **style)
            ax.add_artist(ab)

        # draw not selected points first
        for i, [img, pos] in enumerate(zip(X, Z)):
            if i not in fixed_indices:
                _add_artist(img, pos, frameon=False)

        # draw selected point after to be appeared on top
        highlighted = dict(pad=0.25, frameon=True, bboxprops=dict(edgecolor="#800080"))
        for i in fixed_indices:
            _add_artist(X[i], Z[i], **highlighted)

    fixed_indices, des_pos = list(zip(*fixed_points))

    for i, [ax, Z, stress] in enumerate(zip([ax0, ax1], [Z0, Z1], stresses)):
        _style_axes(ax, show_coordinates=True, hide_ticks=(i > 0))
        _scatter_image(ax, Z, fixed_indices=fixed_indices)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_title(f"Stress: {stress:.2f}")

    _show_moved_points(ax0, Z0[fixed_indices, :], np.array(des_pos), annote_src=False)

    ax2.axis("off")
    ax2.set_title("Legend")

    fig.savefig(out_name, bbox_inches="tight")
