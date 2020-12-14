import json
from functools import partial, lru_cache
from itertools import combinations

import joblib

import dash
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import dash_html_components as html

from server import app

from pmds import pmds_MAP2


STATIC_DIR = "./static"


def get_image_url(dataset_name, img_id, cmap_type="gray"):
    """Return URL for image of a datapoint `img_id` from a stacked svg"""
    return (f"{STATIC_DIR}/{dataset_name}_{cmap_type}.svg#{img_id}",)


def get_image_elem(dataset_name, img_id, cmap_type="gray", width="100px"):
    """Generate dash html image element"""
    return html.Img(
        src=get_image_url(dataset_name, img_id, cmap_type), width=f"{width}"
    )


@lru_cache(maxsize=None)
def get_embedding(dataset_name, method_name="MAP2"):
    in_name = f"{STATIC_DIR}/{dataset_name}_{method_name}.Z"
    return joblib.load(in_name)


def _build_cyto_nodes(dataset_name, cmap_type="gray"):
    Z, _ = get_embedding(dataset_name)
    print("Get embedding: ", Z.shape)
    return [
        dict(
            group="nodes",
            classes="img-node",
            data=dict(
                id=str(idx),
                label=f"node_{idx}",
                url=get_image_url(dataset_name, idx, cmap_type),
            ),
            position=dict(x=x, y=y),
        )
        for idx, [x, y] in enumerate(Z)
    ]


@app.callback(
    Output("cytoplot", "elements"),
    [
        Input("select-dataset", "value"),
        Input("select-cmap", "value"),
    ],
    [
        State("cytoplot", "elements"),
    ],
)
def update_cytoplot(dataset_name, cmap_type, current_elems):
    # update new function of dash 0.38: callback_context
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    if None in [dataset_name, cmap_type]:
        return []

    # always update new nodes (to update img_size, update position or cmap)
    nodes = _build_cyto_nodes(dataset_name, cmap_type)
    edges = []

    # get lastest triggered event (note, in the doc, it takes the first elem)
    # https://dash.plot.ly/faqs
    # we can also access to `ctx.stages` and `ctx.inputs`
    last_event = ctx.triggered[-1]
    last_btn = last_event["prop_id"].split(".")[0]
    print("Last button: ", last_btn)

    return nodes


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