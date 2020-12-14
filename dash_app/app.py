import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto

from server import app
import cytoplot_callbacks


list_datasets = ["digits012", "fmnist", "fmnist_subset"]

###############################################################################
# cytoscape stylesheet
# ref for cytospace js style: http://js.cytoscape.org/
default_cyto_node_style = dict(
    selector=".img-node",
    style={
        "width": 0.04,
        "height": 0.04,
        "shape": ["rectangle", "ellipse"][0],
        "border-color": "white",
        "overlay-opacity": 0.0,
        "background-color": "white",
        "background-fit": ["contain", "cover"][0],
        "background-image": "data(url)",
    },
)

default_cyto_selected_node_style = dict(
    # auto supported selector: http://js.cytoscape.org/#selectors/state
    selector="node:selected",
    style={"shape": "ellipse", "border-width": 0.01, "border-color": "blue"},
)


additional_cyto_css = []


###############################################################################
# layout components

# INLINE = {"display": "inline-block"}
buttons_layout = {
    "btn-submit": ("Update Viz", "secondary"),
}
control_buttons = html.Div(
    [
        dbc.Button(
            id=btn_id,
            children=label,
            n_clicks_timestamp=0,
            outline=True,
            color=color,
            style={"padding": "4px", "margin": "4px"},
        )
        for btn_id, (label, color) in buttons_layout.items()
    ],
    className="mr-3",
)

cytoplot_layout = cyto.Cytoscape(
    id="cytoplot",
    layout={"name": "preset", "animate": True, "fit": True},
    style={"width": "100%", "height": "90vh"},
    stylesheet=[
        default_cyto_node_style,
        default_cyto_selected_node_style,
    ]
    + additional_cyto_css,
    elements=[],
    # autoungrabify=True,  # can not move nodes
    # autounselectify=False,  # can select nodes
)

debug_layout = html.Pre(
    id="txt-debug",
    children="Debug",
    style={"display": "inline", "overflow": "scroll", "border": "1px solid #ccc"},
)


###############################################################################
# local storage for storing links
# links_storage_memory = dcc.Store(id="links-memory", storage_type="memory")
# links_storage_memory_debug = dcc.Store(id="links-memory-debug", storage_type="memory")


##############################################################################
# control components in the navbar

select_dataset_name = dbc.FormGroup(
    [
        dbc.Label("Dataset name", html_for="select-dataset", className="mr-2"),
        dcc.Dropdown(
            id="select-dataset",
            value=None,
            options=[{"label": name, "value": name} for name in list_datasets],
            style={"width": "160px"},
        ),
    ],
    className="mr-3",
)

option_select_scatter_color = dbc.FormGroup(
    [
        dbc.Label("Color item", className="mr-2"),
        dcc.Dropdown(
            id="select-cmap",
            value="gray",
            options=[
                {"label": label, "value": value}
                for label, value in [
                    ("Gray", "gray"),
                    # ("Gray invert", "gray_r"),
                    ("Color", "color"),
                ]
            ],
            style={"width": "120px"},
        ),
    ],
    className="mr-3",
)

slider_select_scatter_zoom_factor = dbc.FormGroup(
    [
        dcc.Slider(
            id="slider-img-size",
            min=0.0,
            max=1.5,
            step=0.2,
            value=0.5,
            included=False,
            marks={i * 0.1: f"{i*0.1:.1f}" for i in range(1, 16, 1)},
        )
    ],
    className="mr-3",
)

data_control_form = dbc.Form([select_dataset_name], inline=True, style={"width": "30%"})

zoom_control_form = dbc.Form(
    [slider_select_scatter_zoom_factor], style={"width": "30%"}
)

display_control_form = dbc.Form(
    [
        option_select_scatter_color,
    ],
    inline=True,
    style={"width": "40%"},
)

navbar = dbc.Navbar(
    [data_control_form, zoom_control_form, display_control_form],
    style={"width": "100%"},
)

##############################################################################
# main app layout
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                navbar,
                # links_storage_memory,
                # links_storage_memory_debug,
            ]
        ),
        dbc.Row(
            [
                dbc.Col([cytoplot_layout], md=10),
                dbc.Col([control_buttons], md=2),
            ]
        ),
    ],
    fluid=True,
)


if __name__ == "__main__":
    app.run_server(debug=True, threaded=True, host="0.0.0.0", processes=1)