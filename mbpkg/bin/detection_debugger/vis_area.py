from typing import Tuple

from dash import html, callback, Input, Output, State, dcc, html, no_update, ctx
import dash_bootstrap_components as dbc
import plotly.express as px

from mbpkg.movie_loading import Movie
from mbpkg.detector import Detector

from .imp_modal import gen_importation_modal_body
from .debug_detection import debug_detector

################## Constants ##################

VISU_LAYERS = ["cells", "midbody", "sir", "s_layer", "s_cube"]

################## Callbacks ##################

@callback(
    Output("vis_imp_modal", "is_open"),
    Input("vis_imp_btn", "n_clicks"),
    State("vis_imp_modal", "is_open")
)
def open_modal(n_clicks, is_open):
    if n_clicks:
        return not is_open

@callback(
    Output("vis_ppl_override", "children"),
    Input("vis_ppl_dd", "value"),
)
def update_ppl_override(detector_repr: str):    
    if not isinstance(detector_repr, str) or len(detector_repr) == 0:
        return no_update
    
    sel_detector = Detector(detector_repr)
    kind = sel_detector.get_spot_method_kind()

    if kind == Detector.SpotMethodKind.bigfish:
        return "big fish parameter tweaking not supported"
    elif kind == Detector.SpotMethodKind.h_maxima:
        return "h_maxima parameter tweaking not supported"
    elif kind == Detector.SpotMethodKind.log:
        return gen_log_doh_override(sel_detector)
    elif kind == Detector.SpotMethodKind.dog:
        return gen_dog_override(sel_detector)
    elif kind == Detector.SpotMethodKind.doh:
        return gen_log_doh_override(sel_detector)
    else:
        raise RuntimeError(f"Unsupported spot method kind: {kind}")


@callback(
    Output("vis_graph_frame", "value"),
    Output("vis_graph_frame", "max"),
    Input("vis_graph_lbtn", "n_clicks"),
    Input("vis_graph_rbtn", "n_clicks"),
    Input("sig_import_movie_out", "data"),
    State("vis_graph_frame", "value"),
)
def update_graph_frame_input(l_clicks: int, r_clicks: int, data: str, cur_frame_idx: int):
    from .shared import _movie

    # if the entered value is out of bounds...
    local_cur_frame_idx = cur_frame_idx if isinstance(cur_frame_idx, int) else 0
    
    if ctx.triggered_id == "vis_graph_lbtn" and l_clicks > 0:
        return max(0, local_cur_frame_idx - 1), no_update
    
    elif ctx.triggered_id == "vis_graph_rbtn" and r_clicks > 0:
        return min(local_cur_frame_idx + 1, _movie.get_max_frame_idx()), no_update
    
    elif ctx.triggered_id == "sig_import_movie_out" and isinstance(_movie, Movie):
        return 0, _movie.get_max_frame_idx()
    
    else:
        return 0, 0 # at init, set max and frame at 0
    

@callback(
    Output("vis_graph_maxframe", "children"),
    Input("sig_import_movie_out", "data")
)
def update_graph_maxframe(data: str):
    from .shared import _movie
    if isinstance(data, str) and len(data) > 0 and isinstance(_movie, Movie):
        return f"/{_movie.get_max_frame_idx()}"
    else:
        return "/0"

@callback(
    Output("vis_graph", "figure"),
    Input("sig_import_movie_out", "data"),
    Input("vis_graph_frame", "value"),
    Input("vis_layer_dd", "value"),

    Input("vis_ppl_p_min_sigma",   "value"),
    Input("vis_ppl_p_max_sigma",   "value"),
    Input("vis_ppl_p_num_sigma",   "value"),
    Input("vis_ppl_p_sigma_ratio", "value"),
    Input("vis_ppl_p_threshold",   "value"),

    State("vis_ppl_dd", "value")
)
def update_graph(
        out_data: str,
        frame: int,
        layer: str,

        min_sigma: float,
        max_sigma: float,
        num_sigma: float,
        sigma_ratio: float,
        threshold: float,

        detector_repr: str,
        ) -> dict:
    
    from .shared import _movie
    
    # most sanity checks
    # the pipeline parameters will have to be checked later, based on detector_repr
    if (not isinstance(out_data, str) 
        or not isinstance(_movie, Movie)
        or not isinstance(frame, int)
        or not isinstance(layer, str)
        or not isinstance(detector_repr, str)): 
        return no_update
    
    layer_idx = {
        "cells":   2,
        "sir":     0,
        "midbody": 1,
        "s_layer": 2,
        "s_cube":  2,
    }

    if layer in ["cells", "sir", "midbody"]:
        return px.imshow(_movie.get_any_layer(layer_idx[layer], frame))
    elif layer in ["s_layer", "s_cube"]:
        mb_img = _movie.get_mklp_layer(frame)
        
        return debug_detector(
            mb_img,
            detector_repr, 
            {
                "min_sigma": min_sigma,
                "max_sigma": max_sigma,
                "num_sigma": num_sigma,
                "sigma_ratio": sigma_ratio,
                "threshold": threshold
            },
            layer
        )
    else:
        raise RuntimeError(f"Unknown layer {layer}")

################# Helpers #################



################## Layout ##################

def gen_left_col(width: int, detectors: list[Detector]) -> dbc.Col:
    return dbc.Col([
        gen_detection_choice_row("30vh", detectors),
        gen_detection_override_row("50vh"),
        gen_detection_spots_row("20vh"),
    ], width=width)

def gen_detection_choice_row(height: str, detectors: list[Detector]) -> dbc.Row:
    return dbc.Row([
        dcc.Dropdown(
            id="vis_ppl_dd",
            options=[d.repr for d in detectors],
            value=detectors[0].repr,
        )
    ], style={"height": height})

def gen_detection_override_row(height: str) -> dbc.Row:
    return dbc.Row([
        html.Div(
            id="vis_ppl_override",
            children="[override row here]"
        ),
    ], style={"height": height})

def gen_log_doh_override(sel_detector: Detector) -> list:
    kws = sel_detector.get_partial().keywords
    min_sigma = kws["min_sigma"]
    max_sigma = kws["max_sigma"]
    num_sigma = kws["num_sigma"]
    threshold = kws["threshold"]
    return [
        dbc.InputGroup([dbc.InputGroupText("min sigma"),   dbc.Input(id="vis_ppl_p_min_sigma", type="number", value=min_sigma)]),
        dbc.InputGroup([dbc.InputGroupText("max sigma"),   dbc.Input(id="vis_ppl_p_max_sigma", type="number", value=max_sigma)]),
        dbc.InputGroup([dbc.InputGroupText("sigma count"), dbc.Input(id="vis_ppl_p_num_sigma", type="number", value=num_sigma)]),
        html.Div(dbc.InputGroup([dbc.InputGroupText("sigma ratio"), dbc.Input(id="vis_ppl_p_sigma_ratio", type="number", value=0)]), style={"display": "none"}),
        dbc.InputGroup([dbc.InputGroupText("threshold"),   dbc.Input(id="vis_ppl_p_threshold", type="number", value=threshold)]),
    ]

def gen_dog_override(sel_detector: Detector) -> list:
    kws = sel_detector.get_partial().keywords
    min_sigma   = kws["min_sigma"]
    max_sigma   = kws["max_sigma"]
    sigma_ratio = kws["sigma_ratio"]
    threshold   = kws["threshold"]
    return [
        dbc.InputGroup([dbc.InputGroupText("min sigma"),   dbc.Input(id="vis_ppl_p_min_sigma", type="number", value=min_sigma)]),
        dbc.InputGroup([dbc.InputGroupText("max sigma"),   dbc.Input(id="vis_ppl_p_max_sigma", type="number", value=max_sigma)]),
        html.Div(dbc.InputGroup([dbc.InputGroupText("sigma count"), dbc.Input(id="vis_ppl_p_num_sigma", type="number", value=0)]), style={"display": "none"}),
        dbc.InputGroup([dbc.InputGroupText("sigma ratio"), dbc.Input(id="vis_ppl_p_sigma_ratio", type="number", value=sigma_ratio)]),
        dbc.InputGroup([dbc.InputGroupText("threshold"),   dbc.Input(id="vis_ppl_p_threshold", type="number", value=threshold)]),
    ]

def gen_detection_spots_row(height: str) -> dbc.Row:
    return dbc.Row([
        html.Div(
            id="vis_spot_list",
            children="[Spot list here]"
        )
    ], style={"height": height})

def gen_center_col(width: int) -> dbc.Col:
    return dbc.Col([
        gen_graph_row("90vh"),
        gen_graph_control_row("10vh")
    ], width=width)

def gen_graph_row(height: str) -> dbc.Row:
    return dbc.Row([
        dcc.Graph(
            id="vis_graph",
            figure={}
        )
    ], style={"height": height})

def gen_graph_control_row(height: str) -> dbc.Row:
    return dbc.Row([
        dbc.Col(html.Div(dbc.Button(
            id="vis_graph_lbtn",
            children="<",
            outline=True,
            color="secondary"
        ), className="d-grid gap-2"), width=2),
        dbc.Col(dbc.InputGroup([
            dbc.InputGroupText("frame:"),
            dbc.Input(id="vis_graph_frame", type="number", value=0, min=0, max=0),
            dbc.InputGroupText(id="vis_graph_maxframe", children="/0"),
        ]), width=8),
        dbc.Col(html.Div(dbc.Button(
            id="vis_graph_rbtn",
            children=">",
            outline=True,
            color="secondary"
        ), className="d-grid gap-2"), width=2),
    ], style={"height": height})

def gen_right_col(width: int, dir_dd_options: list[str]) -> dbc.Col:
    return dbc.Col([
        gen_movie_imp_row("15vh", dir_dd_options),
        gen_layer_control_row("45vh"),
        gen_visualization_control_row("40vh")
    ], width=width)

def gen_movie_imp_row(height: str, dir_dd_options: list[str]) -> dbc.Row:
    return dbc.Row([
        dbc.Button(
            id="vis_imp_btn",
            children="Pick a file",
            color="primary",
            outline=True,
        ),
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Choose a mitosis file")),
            dbc.ModalBody(gen_importation_modal_body(dir_dd_options)),
        ], is_open=False, id="vis_imp_modal", size="xl"),

    ], style={"height": height})

def gen_layer_control_row(height: str) -> dbc.Row:
    return dbc.Row([
        dcc.Dropdown(
            id="vis_layer_dd",
            options=VISU_LAYERS,
            value=VISU_LAYERS[0]
        ),
        html.Div(
            id="vis_layer_settings",
            children="[Additional layer settings]"
        )
    ], style={"height": height})

def gen_visualization_control_row(height: str) -> dbc.Row:
    return dbc.Row([
        "visualization control row"
    ], style={"height": height})


def gen_visualization_area(dir_dd_options: list[str], detectors: list[Detector]):
    return dbc.Container([dbc.Row([
        gen_left_col(1, detectors),
        gen_center_col(10),
        gen_right_col(1, dir_dd_options)
    ])])


# def gen_visualization_area(dir_dd_options):
#     return dbc.Container([ dbc.Row([

#         dbc.Col(
#             children=[
#                 dbc.Row(dbc.Col([
#                     dbc.Alert("Difference of Gaussian", color="secondary"),
#                     html.Div(dbc.InputGroup([dbc.InputGroupText("Min:"), dbc.Input(placeholder="B1", type="number")])),
#                     html.Div(dbc.InputGroup([dbc.InputGroupText("Max:"), dbc.Input(placeholder="B1", type="number")])),
#                     html.Div(dbc.InputGroup([dbc.InputGroupText("N:"), dbc.Input(placeholder="B1", type="number")])),
#                 ], style={"height": "50vh"})),

#                 dbc.Row(dbc.Col([
#                     dbc.Button(
#                         id="open-modal",
#                         children="open modal"
#                     ),
#                     dbc.Modal([
#                         dbc.ModalHeader(dbc.ModalTitle("Chose another file")),
#                         dbc.ModalBody(gen_importation_modal_body(dir_dd_options)),
#                     ], is_open=False, id="my-modal", size="xl")
#                 ], style={"height": "50vh"})),
#             ], 
#             width=2
#         ),

#         dbc.Col(
#             children=["WIP"],
#             width=8),

#         dbc.Col(
#             children=[
#                 dbc.Row(dbc.Col(children=[
#                     dbc.Alert("Determinant of Hessian", color="success"),
#                     html.Div(["min", dbc.Input(placeholder="C1", type="number")]),
#                     html.Div(["max", dbc.Input(placeholder="C2", type="number")]),
#                     html.Div(["n", dbc.Input(placeholder="C3", type="number")]),
#                 ], style={"height": "50vh"})),
#                 dbc.Row(dbc.Col(children=[
#                     dbc.Alert("Determinant of Hessian", color="success"),
#                     html.Div(["min", dbc.Input(placeholder="C1", type="number")]),
#                     html.Div(["max", dbc.Input(placeholder="C2", type="number")]),
#                     html.Div(["n", dbc.Input(placeholder="C3", type="number")]),
#                 ], style={"height": "50vh"})),
#             ],
#             width=2
#         )
#     ])], fluid=True)