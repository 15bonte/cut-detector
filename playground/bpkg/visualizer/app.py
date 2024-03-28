import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.express as px

import os
from dash import Dash, html, dash_table, dcc, callback, Output, Input
from typing import List

from ..main import TEST_FILE_DIR
from ..data_loader import load_movie, Movie
from .nparray_encdec import np_array_encode, np_array_decode


from skimage import data

# Initialize the app
print("__name__:", __name__)
app = Dash(__name__)
app.title = "Blob Visualizer"

# Color theme
colors = {
    "background": "#111111",
    "title": "#FFFFFF",
    "text": "#7FDBFF",
}


app.layout = dbc.Container(style={'backgroundColor': colors['background'], "height":"100vh"}, children=[
    html.H1(
        children="Blob Detection and Tracking Visualizer",
        style={
            'textAlign': 'center',
            'color': colors['title']
        }
    ),
    html.Hr(
        style={"borderColor": "red", "color": "red", "width": "100%"}
    ),
    html.H2(
        children = "Importation Settings",
        style={
            'textAlign': 'left',
            'color': colors['text']
        }
    ),
    # dbc.Button("Primary", color="primary", id="button_2"),
    # dcc.Graph(id="test_app_figure_2"),
    dbc.Row([ 
        html.Div([
            dbc.InputGroup([
                dbc.InputGroupText("Source File directory", style={'color': colors['text']}),
                dbc.Input(id="source_dir_field", value=TEST_FILE_DIR, placeholder="Source directory"),
            ]),
            dcc.Dropdown(id="source_choice_dropdown"),
            dbc.Alert(id="source_dir_state_box", children="Waiting for directory", color="info"),
            dcc.RadioItems(id="source_format", options=["4 channels", "long 3 channels"], value="4 channels"),
        ]), 
    ]),
    dcc.Store(id="movie_data"),
    html.Hr(
        style={"borderColor": "red", "color": "red", "width": "100%"}
    ),
    html.H2(
        children = "Detection",
        style={
            'textAlign': 'left',
            'color': colors['text']
        }
    ),
    dbc.Row([
        dbc.InputGroup([
            dbc.InputGroupText("Detection Method used", style={'color': colors['text']}),
            dcc.RadioItems(id="detection_method", options=["LoG", "DoG", "DoH"], value="LoG", inputStyle={"marginLeft": "100px"}, inline=True),
        ]),
    ]),
    dbc.Row(id="detection_method_settings"),
    html.Hr(
        style={"borderColor": "red", "color": "red", "width": "100%"}
    ),
    dcc.Graph(id="detection_graph"),
    dcc.Slider(min=0, max=0, step=1, value=0, id="detection_channel_slider"),
    dcc.Slider(min=0, max=0, step=1, value=0, id="detection_frame_slider",
               tooltip={"placement": "bottom", "always_visible": True}),
], fluid=True)

@callback(
    Output("source_choice_dropdown", "options"),
    Output("source_dir_state_box", "children"),
    Output("source_dir_state_box", "color"),
    Input("source_dir_field", "value")
)
def update_source_choice_dropdown(value):
    try:
        children = ["Directory Found, you can choose the source file"]
        color = "success"
        return [f.path for f in os.scandir(value) if f.is_file()], children, color
    except FileNotFoundError:
        children = [f"Directory {value} not found"]
        color = "danger"
        print("invalid directory: ", value)
        return [], children, color

@callback(
    Output("movie_data", "data"),
    Input("source_choice_dropdown", "value"),
    Input("source_format", "value")
)
def load_movie_data(path: str | None, source_format: str | None):
    print(f"load_movie_data path:{path} source_format:{source_format}")
    if path is None or source_format is None:
        return "0"
    d = {
        "4 channels": "4c",
        "long 3 channels": "3c",
    }
    movie = load_movie(path, d[source_format])
    return np_array_encode(movie)

@callback(
    Output("detection_method_settings", "children"),
    Input("detection_method", "value")
)
def update_detection_method_settings(method: str):
    # print("detection method:", method)
    if method == "LoG" or method == "DoH":
        return [
            dbc.Col([gen_input_group(c, "", "number", f"detection_method_{c}")], width=3) 
            for c in ["max_sigma", "min_sigma", "num_sigma", "threshold"]
        ]

    elif method == "DoG":
        return [
            dbc.Col([gen_input_group(c, "", "number", f"detection_method_{c}")], width=3) 
            for c in ["max_sigma", "min_sigma", "sigma_ratio", "threshold"]
        ]

@callback(
    Output("detection_channel_slider", "max"),
    Output("detection_frame_slider", "max"),
    Input("movie_data", "data")
)
def update_detection_sliders_from_movie_data(movie_data: str):
    # if the data is available, we have a json dict with at least the following keys:
    # dtype/shape/data
    # so str len, without real data, spaces, etc... must be at least
    # size 5+5+4=14
    if len(movie_data) < 14:
        return 0, 0
    
    movie_arr = np_array_decode(movie_data)
    # Not great, because right now, nothing makes sure the selected file
    # is a movie (it could be any TIFF)
    # Though in reality, we can't really differentiate 2 np arrays TYXC from
    # each other...
    movie = Movie.make_movie_from_array(movie_arr) 
    framecount = movie.get_framecount()
    layercount = movie.get_layercount()
    return layercount, framecount

# @callback(
#     # Output("detection_graph", "figure"),
#     Output("test_app_figure_2", "figure"),
#     Input("movie_data", "data"),
#     Input("detection_channel_slider", "value"),
#     Input("detection_frame_slider", "value"),
# )
# def update_detection_graph(movie_data: str, chan_slider: int, frame_slider: int):
#     print("update_detection_graph called")
#     # Movie readiness check
#     if len(movie_data) < 14:
#         return {}

#     data = load_movie(
#     "./src/cut_detector/data/mid_bodies_movies_test/s4_siLuci-1_mitosis_24_128_to_135.tiff",
#     "4c"
#     )
#     mv = Movie.make_movie_from_array(data)
#     img = mv.get_layer_frame(1, 0)

#     # img = data.chelsea()
#     fig = px.imshow(img)
#     print(fig)
#     return fig


@callback(
    Output("detection_graph", "figure"),
    Input("movie_data", "data"),
    Input("detection_channel_slider", "value"),
    Input("detection_frame_slider", "value"),
)
def update_detection_graph(movie_data, chan_idx, frame_idx):
    if len(movie_data) < 14 \
        or not isinstance(chan_idx, int) \
        or chan_idx < 0 \
        or not isinstance(frame_idx, int) \
        or frame_idx < 0:
        return {}
    data = np_array_decode(movie_data)
    mv = Movie.make_movie_from_array(data)
    img = mv.get_layer_frame(frame_idx, chan_idx)

    # img = data.chelsea()
    fig = px.imshow(img)
    # print(fig)
    return fig


def gen_input_group(pre_text: str, ipt_ph: str, ipt_type: str, ipt_id: str) -> dbc.InputGroup:
    return dbc.InputGroup([
        dbc.InputGroupText(pre_text),
        dbc.Input(placeholder=ipt_ph, type=ipt_type, id=ipt_id)
    ])

def run_app(flags: List[str]):
    print(f"App: running Visualizer app with args: {flags}")
    app.run(debug=True)





