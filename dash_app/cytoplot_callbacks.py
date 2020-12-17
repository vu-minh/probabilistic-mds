import json
import joblib
from functools import lru_cache, partial
from collections import defaultdict

import numpy as np
import dash
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import dash_html_components as html

from server import app
from app_logic import run_pmds


STATIC_DIR = "./static"
IMAGE_DATASETS = ("digits", "fmnist")
TABULAR_DATASETS = ()
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


@lru_cache(maxsize=None)
def get_init_embedding(dataset_name, method_name="MAP2"):
    in_name = f"{STATIC_DIR}/{dataset_name}_{method_name}.Z"
    return joblib.load(in_name)


@lru_cache(maxsize=None)
def _build_cyto_node_image(idx, x, y, dataset_name, cmap_type, color):
    return dict(
        group="nodes",
        classes="img-node",
        data=dict(
            id=str(idx),
            label=f"node_{idx}",
            color=color,
            url=get_image_url(dataset_name, idx, cmap_type),
        ),
        position=dict(x=x, y=y),
    )


@lru_cache(maxsize=None)
def _build_cyto_node_with_label(idx, x, y, dataset_name, cmap_type, color):
    return dict(
        group="nodes",
        classes="label-node",
        data=dict(id=str(idx), label=f"node_{idx}", color=color),
        position=dict(x=x, y=y),
    )


@lru_cache(maxsize=None)
def _build_cyto_node_normal(idx, x, y, dataset_name, cmap_type, color):
    return dict(
        group="nodes",
        classes="normal-node",
        data=dict(id=str(idx), color=color),
        position=dict(x=x, y=y),
    )


def _gen_color_from_label(labels, dataset_name, cmap_type="jet"):
    # import matplotlib as mpl
    # import matplotlib.cm as cm
    from matplotlib.colors import rgb2hex
    from matplotlib.cm import get_cmap

    if dataset_name in ARTIFICIAL_DATASET or len(np.unique(labels)) > 10:
        cmap_type = "jet"
    else:
        cmap_type = "tab10"

    # my_cmap = cm.ScalarMappable(
    #     norm=mpl.colors.Normalize(vmin=0, vmax=1), cmap=cmap_type
    # )
    # return list(map(my_cmap.to_rgba, labels))

    cmap = get_cmap(cmap_type)
    labels = np.array(labels) / np.max(labels)
    return list(map(lambda lbl: rgb2hex(cmap(lbl)), labels))


def _build_cyto_nodes(dataset_name, Z, labels=None, cmap_type="gray"):
    # build different node types according to `dataset_name`
    if dataset_name.startswith(IMAGE_DATASETS):
        build_node_func = _build_cyto_node_image
    elif dataset_name.startswith(TABULAR_DATASETS):
        build_node_func = _build_cyto_node_with_label
    else:
        build_node_func = _build_cyto_node_normal

    # gen color for each node from labels
    labels = range(len(Z)) if labels is None else labels
    colors = _gen_color_from_label(labels, dataset_name, cmap_type)
    return [
        build_node_func(idx, x, y, dataset_name, cmap_type, color)
        for idx, ([x, y], color) in enumerate(zip(Z, colors))
    ]


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
        and selected_nodes is not None
        and current_embedding is None
    ):
        # update viz using user's fixed points
        Z = run_pmds(dataset_name, current_embedding[1], fixed_points=selected_nodes)
        current_embedding[1] = Z
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
