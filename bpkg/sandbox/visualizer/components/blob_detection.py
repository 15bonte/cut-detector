from dash import callback, Input, Output, State, dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px

from typing import List, Dict

import data_loading
from data_loading import Movie
from .detection_components.frame_control import generate_slider_side_buttons

LocalMovie: Movie = None
LocalMovieInfo: str = ""

def render_blob_detection() -> List:
    graph = dcc.Graph("detection-graph", figure={})
    graph_mode_select = dbc.Select(
        id="detection-graph-mode",
        options=["Image", "Norm", "Layer", "3D"],
        value="Image"
    )
    graph_mode_select = dbc.Row([
        dbc.Label("Plotting:", html_for="detection-graph-mode", width=1),
        dbc.Col(
            dbc.Select(
                id="detection-graph-mode",
                options=["Image", "Norm", "Layer", "3D"],
                value="Image"
            ),
            width=11
        )
    ], class_name="mb-3")

    frame_number_input = dbc.InputGroup(
        [
            dbc.InputGroupText("Go To Frame:"),
            dbc.Input(
                id="detection-frame-input", 
                type="number", min=0, max=0, value=0, step=1
            ),
            dbc.InputGroupText(id="detection-frame-input-rgt", children="/ 0 max frame index"),
        ],
        className="mb-3",
    ),

    param_table = dbc.Row([
        generate_blob_detection_col("log", "Laplacian of Gaussian",  4),
        generate_blob_detection_col("dog", "Difference of Gaussian", 4),
        generate_blob_detection_col("doh", "Determinant of Hessian", 4),
    ])

    blob_detection_mode = dbc.Select(
        id="detection-mode",
        options=[
            {"label": "Laplacian of Gaussian", "value": "log"},
            {"label": "Difference of Gaussian", "value": "dog"},
            {"label": "Determinant of Hessian", "value": "doh"},
        ],
        value="log"
    )
    
    l = [
        dbc.Row(graph),
        generate_slider_side_buttons("detection-layer-slider", True),
        generate_slider_side_buttons("detection-frame-slider", False),
        dbc.Row(frame_number_input),

        graph_mode_select,
        param_table,
        dbc.Form(blob_detection_mode),
    ]
    return l

def generate_blob_detection_col(id: str, name: str, width: int) -> dbc.Col:
    def generate_param_line(param_name: str, param_id: str) -> dbc.InputGroup:
        return dbc.InputGroup([
            dbc.InputGroupText(f"{param_name}:"), 
            dbc.Input(id="{id}-{param_id}", type="number"),
        ])
    
    header = dbc.Alert(
        id=f"{id}-header",
        children=name, 
        color="secondary"
    )
    p1 = generate_param_line("Min Sigma", "min-sig")
    p2 = generate_param_line("Max Sigma", "max-sig")
    p3 = generate_param_line("Sigma Count", "n-sig")
    p4 = generate_param_line("Threshold", "threshold")
    
    return dbc.Col([
        html.Div(header),
        html.Div(p1),
        html.Div(p2),
        html.Div(p3),
        html.Div(p4),
    ], width=width)

def release_detection_movie():
    global LocalMovie, LocalMovieInfo
    LocalMovie = None
    LocalMovieInfo = ""

def load_movie(info: str):
    global LocalMovie, LocalMovieInfo
    kind = info[0:2]
    path = info[3:]
    data = data_loading.load_movie(path, kind)
    LocalMovie = Movie(
        data,
        path
    )
    LocalMovieInfo = info


def local_movie_guard(movie_info: str):
    if len(movie_info) != 0 \
        and (movie_info != LocalMovieInfo or LocalMovie is None):
            load_movie(movie_info)


@callback(
    Output("detection-layer-slider", "max"),
    Output("detection-frame-slider", "max"),
    Output("detection-frame-input", "max"),
    Output("detection-frame-input-rgt", "children"),
    Input("movie-info", "data"),
)
def update_sliders_and_num_input(movie_info: str):
    if len(movie_info) == 0:
        return 0, 0
    local_movie_guard(movie_info)

    layer_count = LocalMovie.get_layercount()
    frame_count = LocalMovie.get_framecount()
    max_layer_idx = layer_count - 1
    max_frame_idx = frame_count - 1
    rgt = f"/ {max_frame_idx} max frame index"

    return max_layer_idx, max_frame_idx, max_frame_idx, rgt


@callback(
    Output("detection-graph", "figure"),
    Input("movie-info", "data"),
    Input("detection-layer-slider", "value"),
    Input("detection-frame-slider", "value")
)
def update_graph(movie_info: str, layer_idx: int, frame_idx: int) -> Dict:
    if len(movie_info) == 0:
        return {}
    # if movie_info != LocalMovieInfo or LocalMovie is None:
    #     load_movie(movie_info)
    local_movie_guard(movie_info)
    img = LocalMovie.get_any_layer(layer_idx, frame_idx)
    return px.imshow(img)



#
# Despite its name, 'horror' is actually very good
# use it as a reference
#
# horror = dbc.Row([
#         dbc.Col(children=[
#             dbc.Alert("Laplacian of Gaussian", color="primary"),
#             html.Div([
#                 dbc.InputGroup([dbc.InputGroupText("A1:"), dbc.Input(placeholder="B1", type="number")]),
#                 dbc.InputGroup([dbc.InputGroupText("A2:"), dbc.Input(placeholder="B1", type="number")])
#             ]), 
#             html.Div(["A1", "A2"]), 
#             html.Div(["A1", "A2"])
#         ], width=4),
#         dbc.Col(children=[
#             dbc.Alert("Difference of Gaussian", color="secondary"),
#             html.Div(dbc.InputGroup([dbc.InputGroupText("Min:"), dbc.Input(placeholder="B1", type="number")])),
#             html.Div(dbc.InputGroup([dbc.InputGroupText("Max:"), dbc.Input(placeholder="B1", type="number")])),
#             html.Div(dbc.InputGroup([dbc.InputGroupText("N:"), dbc.Input(placeholder="B1", type="number")])),
#         ], width=4),
#         dbc.Col(children=[
#             dbc.Alert("Determinant of Hessian", color="success"),
#             html.Div(["min", dbc.Input(placeholder="C1", type="number")]),
#             html.Div(["max", dbc.Input(placeholder="C2", type="number")]),
#             html.Div(["n", dbc.Input(placeholder="C3", type="number")]),
#         ], width=4)
#     ])