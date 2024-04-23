from typing import Callable, Any, Literal
from functools import partial

import numpy as np
from dash import dcc, html, callback, Input, Output, State, no_update, ctx
import dash_mantine_components as dmc
import plotly.express as px
import skimage.data

from cut_detector.utils.mb_support import detection

from mbpkg.movie_loading import Movie
from mbpkg.better_detector import Detector

from .viewer_controller import make_viewer_controller
from .viz_pannel import LAYER_LIT
from .visualizer_fn import make_debug_output, mm_log_visualizer, mm_doh_visualizer

######## Local globals ########
detection_debugger: Callable[[np.ndarray, dict[str, Any]], np.ndarray] = None

######## Constants ########

BACKGROUND_MAPPING: dict[LAYER_LIT, ] = {
    "cells":       2,
    "midbody":     1,
    "sir":         0,
}

FactoryCallable    = Callable[[np.ndarray], np.ndarray]
VizualizerCallable = Callable[[dict[str, Any], np.ndarray, dict[str, Any]], np.ndarray]
CALLABLE_TO_VISUALIZER: dict[Callable, Callable] = {
    detection.detect_minmax_log: mm_log_visualizer,
    detection.detect_minmax_dog: mm_log_visualizer,
    detection.detect_minmax_doh: mm_doh_visualizer,
}

######## Callbacks ########

@callback(
    Output("graph", "figure"),
    Input("sig_imp_file", "data"),
    Input("sig_cur_frame", "data"),
    Input("sig_det_param", "data"),
    State("sig_cur_det", "data"),
    Input("sig_viz_param", "data")
)
def update_graph(
        sig_imp: str, 
        frame: int, 
        det_param: dict,
        cur_det: str, 
        viz_param: dict):
    from .shared import _movie
    global detection_debugger

    if not isinstance(frame, int) or not isinstance(_movie, Movie):
        return no_update
    if not isinstance(sig_imp, str) or len(sig_imp) == 0:
        return no_update
    if (layer := viz_param.get("layer")) is None:
        return no_update
    
    if layer in ["cells", "midbody", "sir"]:
        return px.imshow(
            _movie.get_any_layer(BACKGROUND_MAPPING[layer], frame),
            template="plotly_dark",
        )
    
    elif layer == "sigma_layer":
        if not isinstance(cur_det, str) or len(cur_det) == 0:
            return no_update

        detection_debugger = create_detection_debugger(
            cur_det,
            det_param
        )

        if detection_debugger is None:
            return px.imshow(
                skimage.data.binary_blobs(),
                template="plotly_dark",
            )
        else:
            debug_output = make_debug_output(viz_param)
            return px.imshow(
                detection_debugger(
                    img=_movie.get_mklp_layer(frame), 
                    debug_output=debug_output
                ),
                template="plotly_dark",
            )

    else:
        raise ViewerError(f"Unsupported layer {layer}")
    

######## Helpers ########
class ViewerError(Exception):
    pass

def create_detection_debugger(
        det_str: str, 
        detector_args: dict[str, Any]
        ) -> Callable[[np.ndarray, dict[str, Any]], np.ndarray]:
    
    detector = Detector(det_str)
    if (fn := detector.try_to_partial()) is None:
        print("here")
        return None
    if (associated_fn := CALLABLE_TO_VISUALIZER.get(fn.func)) is None:
        print("there")
        return None
    
    return partial(
        associated_fn,
        detector_args=detector_args
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


