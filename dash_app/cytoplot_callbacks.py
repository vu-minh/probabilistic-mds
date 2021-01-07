import json
import joblib
from functools import lru_cache, partial
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.cm import get_cmap

import dash
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import dash_html_components as html

from server import app
from app_logic import run_pmds


STATIC_DIR = "./static"
IMAGE_DATASETS = ("digits", "fmnist")
TABULAR_DATASETS = ("qpcr",)  # "automobile"
ARTIFICIAL_DATASET = (
    ["swiss_roll", "swiss_roll_noise"]
    + ["s_curve", "s_curve_noise"]
    + ["sphere", "sphere_noise"]
)


def get_image_url(dataset_name, img_id, cmap_type="gray"):
    """Return URL for image of a datapoint `img_id` from a stacked svg"""
    return (f"{STATIC_DIR}/{dataset_name}_{cmap_type}.svg#{img_id}",)


def get_image_elem(dataset_name, img_id, cmap_type="gray", width="100px"):
    """Generate dash html image element"""
    return html.Img(
        src=get_image_url(dataset_name, img_id, cmap_type), width=f"{width}"
    )


def get_init_embedding(dataset_name, method_name="MAP2"):
    in_name = f"{STATIC_DIR}/{dataset_name}_{method_name}.Z"
    return joblib.load(in_name)


@lru_cache(maxsize=None)
def _build_cyto_node_image(idx, x, y, dataset_name, cmap_type):
    return dict(
        group="nodes",
        classes="img-node",
        data=dict(id=str(idx), url=get_image_url(dataset_name, idx, cmap_type)),
        position=dict(x=x, y=y),
    )


@lru_cache(maxsize=None)
def _build_cyto_node_with_label(idx, x, y, color, text_label):
    return dict(
        group="nodes",
        classes="label-node",
        data=dict(id=str(idx), label=f"{idx}-{text_label}", color=color),
        position=dict(x=x, y=y),
    )


@lru_cache(maxsize=None)
def _build_cyto_node_normal(idx, x, y, color):
    return dict(
        group="nodes",
        classes="normal-node",
        data=dict(id=str(idx), color=color),
        position=dict(x=x, y=y),
    )


def _gen_color_from_label(labels, dataset_name, cmap_type="tab10"):
    if dataset_name in ARTIFICIAL_DATASET or len(np.unique(labels)) > 10:
        cmap_type = "Spectral"
    print(f"[DEBUG] using {cmap_type} for {dataset_name}")

    cmap = get_cmap(cmap_type)
    labels = np.array(labels) / np.max(labels)
    return list(map(lambda lbl: to_hex(cmap(lbl)), labels))


def _build_cyto_nodes(dataset_name, Z, labels=None, cmap_type="gray"):
    if labels is None:
        labels = np.zeros(len(Z))
    colors = _gen_color_from_label(labels, dataset_name)

    nodes = []
    for idx, [x, y] in enumerate(Z):
        y = -y  # cytoscape use (0, 0) at the top-left

        if dataset_name.startswith(TABULAR_DATASETS):
            node = _build_cyto_node_with_label(idx, x, y, colors[idx], labels[idx])
        elif dataset_name.startswith(IMAGE_DATASETS):
            node = _build_cyto_node_image(idx, x, y, dataset_name, cmap_type)
        else:  # ARTIFICIAL_DATASET
            node = _build_cyto_node_normal(idx, x, y, colors[idx])

        nodes.append(node)

    return nodes


def debug_embedding(old_Z, new_Z, selected_points=[], colors=None):
    print("[DEBUG]Plot debug selected points: ", selected_points)
    fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(10, 5))

    Z0 = old_Z[selected_points, :]
    Z1 = new_Z[selected_points, :]

    ax0.set_aspect("equal")
    ax0.scatter(*old_Z.T, c=colors)
    ax0.scatter(x=Z0[:, 0], y=Z0[:, 1], marker="+", s=128)
    ax0.scatter(x=Z1[:, 0], y=Z1[:, 1], marker="o", s=128, alpha=0.3)

    ax1.set_aspect("equal")
    ax1.scatter(*new_Z.T, c=colors)
    # ax1.scatter(*old_Z.T, c=None, alpha=0.05)
    # ax1.scatter(x=Z0[:, 0], y=Z0[:, 1], marker="o", s=128, alpha=0.05)
    ax1.scatter(x=Z1[:, 0], y=Z1[:, 1], marker="+", s=128)

    fig.savefig("test_Z.png", bbox_inches="tight")


@app.callback(
    [Output("cytoplot", "elements"), Output("embedding-memory", "data")],
    [
        Input("select-dataset", "value"),
        Input("select-cmap", "value"),
        Input("btn-submit", "n_clicks_timestamp"),
    ],
    [
        State("embedding-memory", "data"),
        State("selected-nodes-memory", "data"),
    ],
)
def update_cytoplot(dataset_name, cmap_type, _, current_embedding, selected_nodes):
    # update new function of dash 0.38: callback_context
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    if None in [dataset_name, cmap_type]:
        return [], None

    # get lastest triggered event (note in the doc, it takes the first elem)
    last_event = ctx.triggered[-1]
    last_btn = last_event["prop_id"].split(".")[0]

    if last_btn in ["select-dataset", "select-cmap"]:
        # simple update cytoplot
        if current_embedding is None or current_embedding[0] != dataset_name:
            Z, _, labels = get_init_embedding(dataset_name)
            current_embedding = [dataset_name, Z, labels]
    elif (
        last_btn == "btn-submit"
        # and selected_nodes is not None
        and current_embedding is not None
    ):
        # update viz using user's fixed points
        selected_nodes = selected_nodes or dict()
        _, old_embedding, labels = current_embedding
        Z = run_pmds(dataset_name, old_embedding, fixed_points=selected_nodes)
        current_embedding[1] = Z
        debug_embedding(
            old_Z=np.array(old_embedding),
            new_Z=np.array(Z),
            selected_points=list(map(int, selected_nodes.keys())),
            colors=_gen_color_from_label(labels, dataset_name),
        )
    else:
        print("[DASH APP] Unexpected event: ", last_btn)
        return [], None

    nodes = _build_cyto_nodes(*current_embedding, cmap_type)
    return nodes, current_embedding


@app.callback(
    Output("cytoplot", "stylesheet"),
    [Input("slider-img-size", "value")],
    [State("cytoplot", "stylesheet")],
)
def change_cyto_style(img_size, current_styles):
    style_list = current_styles
    if img_size:
        scale_factor = {
            ".img-node": 0.05,
            "node:selected": 0.1,
        }
        for style in style_list:
            selector = style["selector"]
            if selector in scale_factor.keys():
                scaled_size = img_size * scale_factor[selector]
                style["style"]["width"] = scaled_size
                style["style"]["height"] = scaled_size
                style["style"]["border-width"] = 0.1 * scaled_size
    return style_list


@app.callback(
    [Output("txt-debug", "children"), Output("selected-nodes-memory", "data")],
    Input("cytoplot", "tapNode"),
    State("selected-nodes-memory", "data"),
)
def store_moved_nodes(tap_node, selected_nodes):
    if tap_node is None:
        return "No tap node", defaultdict(list)

    # store selected nodes in a dict, keyed by "id" which is str type
    node_id, pos = tap_node["data"]["id"], tap_node["position"]
    selected_nodes[node_id] = [pos["x"], pos["y"]]

    debug_info = f"TAP: {node_id} @ {pos}\n"
    debug_info += json.dumps(selected_nodes, indent=2)
    print("SELECTED NODES: ", selected_nodes)

    return debug_info, selected_nodes
