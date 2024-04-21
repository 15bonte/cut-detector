import numpy as np
from dash import dcc, html, callback, Input, Output, no_update
import dash_mantine_components as dmc
import plotly.express as px
import skimage.data

from mbpkg.movie_loading import Movie

from .viewer_controller import make_viewer_controller

######## Constants ########


######## Callbacks ########

@callback(
    Output("graph", "figure"),
    Input("sig_imp_file", "data"),
    Input("sig_cur_frame", "data"),
)
def update_graph(sig_imp: str, frame: int):
    from .shared import _movie

    if not isinstance(frame, int) or not isinstance(_movie, Movie):
        return no_update
    if not isinstance(sig_imp, str) or len(sig_imp) == 0:
        return no_update
    
    return px.imshow(
        _movie.get_mklp_layer(frame),
        template="plotly_dark",
    )


######## Layout ########

def make_viewer() -> dmc.Paper:
    return dmc.Paper(
        children=[
            html.Div(dcc.Graph(id="graph", figure=px.imshow(skimage.data.binary_blobs(), template="plotly_dark"), style={"height": "90vh"})),
            html.Div(make_viewer_controller(), style={"height": "10vh"})
        ],
        style={"height": "100vh"},
        withBorder=True,
    )