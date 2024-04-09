from contextlib import suppress
import sys
import time
from typing import Optional, Tuple

import numpy as np
import plotly.express as px
from dash import Dash, html, dcc, callback, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc

from data_loading import Source, Movie
from detection_gt import GTContainer, GTPoint

####### Global Data #########
Src: Optional[Source] = None
OutFp: Optional[str] = None
Data: np.ndarray = None # Not realy used anymore
MovieData: Movie = None
GroundTruth: dict[int, set[Tuple[int, int]]] = {} # dict[frame_idx, set[point x/y]]

###### Dash Config #########
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)
app.title = "Ground Truth Editor"





###### Layout Functions
def generate_point_state_row() -> dbc.Row:
    plus_button = dbc.Button(
        id="update-gt-btn",
        children="V",
        outline=True,
        color="success",
        class_name="me-1"
    )
    gt_point_dd = dcc.Dropdown(
        id="gt-point-dd",
    )
    minus_button = dbc.Button(
        id="remove-gt-btn",
        children="X",
        outline=True,
        color="danger",
        class_name="me-1"
    )
    return dbc.Row([
        dbc.Col(minus_button, width=1, class_name="d-grid gap-2"),
        dbc.Col(gt_point_dd, width=10, class_name="d-grid gap-2"),
        dbc.Col(plus_button, width=1, class_name="d-grid gap-2"),
    ])

def generate_graph_row() -> dbc.Row:
    return dbc.Row([
        dbc.Col(dcc.Graph(id="graph", figure={}, style={'height': '75vh'}), width=11),
        dbc.Col(html.Div(id="point-summary-row"), width=1)
    ])

def generate_graph_state_row() -> dbc.Row:
    x_coord_ipt =  dbc.InputGroup(
        [
            dbc.InputGroupText("x:"),
            dbc.Input(id="x-gt-input", type="number", min=0),
        ],
        className="mb-3",
    ),
    y_coord_ipt =  dbc.InputGroup(
        [
            dbc.InputGroupText("y:"),
            dbc.Input(id="y-gt-input", type="number", min=0),
        ],
        className="mb-3",
    ),
    return dbc.Row([
        dbc.Col(x_coord_ipt),
        dbc.Col(y_coord_ipt)
    ])

def generate_frame_ctrl_row() -> dbc.Row:
    left_button = dbc.Button(
        id="left-frame-btn",
        children="<",
        color="primary",
        class_name="me-1"
    )
    frame_input =  dbc.InputGroup(
        [
            dbc.InputGroupText("Frame"),
            dbc.Input(
                id="frame-input", 
                type="number", 
                value=0,
                min=0),
            dbc.InputGroupText(f"/0"),
        ],
        className="mb-3",
    ),
    right_button = dbc.Button(
        id="right-frame-btn",
        children=">",
        color="primary",
        class_name="me-1"
    )
    return dbc.Row([
        dbc.Col(left_button, width=1, class_name="d-grid gap-2"),
        dbc.Col(frame_input, width=10),
        dbc.Col(right_button, width=1, class_name="d-grid gap-2"),
    ])

def generate_save_row() -> dbc.Row:
    return dbc.Row([
        dbc.Button(
            id="save-bt",
            children="save ground truth",
            color="success",
            class_name="me-1"
        )
    ])

def generate_point_summary_row() -> dbc.Row:
    return dbc.Row(children=[])
    # return dbc.Row(id="point-summary-row", children=[])

app.layout = dbc.Container([
    generate_point_state_row(),
    generate_graph_row(),
    generate_graph_state_row(),
    generate_frame_ctrl_row(),
    generate_save_row(),
    generate_point_summary_row(),
])







###### Callbacks ########

@callback(
        Output("graph", "figure"),
        Input("frame-input", "value")
)
def update_graph_with_frame(frame):
    if not isinstance(frame, int):
        return no_update
    
    max_frame = MovieData.get_max_frame_idx()
    return px.imshow(MovieData.get_mklp_layer(
        max(0, min(frame, max_frame))
    ))


@callback(
        Output("frame-input", "value"),
        Input("left-frame-btn", "n_clicks"),
        Input("right-frame-btn", "n_clicks"),
        State("frame-input", "value")
)
def update_frame_with_button(
        left_click: int, 
        right_click: int, 
        frame_input: Optional[int]):
    
    if frame_input is None or ctx.triggered_id is None:
        return 0

    btn_clicked = ctx.triggered_id
    if btn_clicked == "left-frame-btn":
        # return max(0, frame_input-1)
        new_value = frame_input-1
    elif btn_clicked == "right-frame-btn":
        # return min(frame_input+1, MovieData.get_max_frame_idx())
        new_value = frame_input+1
    else:
        raise RuntimeError(f"Unknown button id:{btn_clicked}")
    
    return max(0, min(new_value, MovieData.get_max_frame_idx()))

@callback(
        Output("x-gt-input", "value"),
        Output("y-gt-input", "value"),
        Input("graph", "clickData")
)
def update_xy_with_graph(click_data):
    if click_data is None:
        return no_update, no_update
    x,y = get_xy_from_click_data(click_data)
    return x, y


@callback(
        Output("gt-point-dd", "options"),
        Input("graph", "clickData"),
        Input("frame-input", "value"),
        Input("update-gt-btn", "n_clicks"),
        Input("remove-gt-btn", "n_clicks"),
        State("gt-point-dd", "options"),
        State("x-gt-input", "value"),
        State("y-gt-input", "value"),
        State("gt-point-dd", "value"),
)
def update_ground_truth_dd(
        click_data: dict, 
        frame: int,
        update_clicks: int,
        remove_clicks: int,
        dd_options: list[str],
        x_ipt: int,
        y_ipt: int,
        dd_value: str):
    
    trigger_id = ctx.triggered_id

    if trigger_id == "graph" and isinstance(click_data, dict):
        # new point clicked
        # - add point to dict
        # - add line to dropdown
        # /!\ don't use x/y: we cannot assume x/y callback has run before this one
        x, y = get_xy_from_click_data(click_data)
        frame_idx = valid_frame_idx(frame)
        insert_ground_truth(frame_idx, x, y)

        new_options = dd_options if isinstance(dd_options, list) else []
        new_options.append(f"x:{x} y:{y}")
        return new_options


    elif trigger_id == "frame-input":
        # frame has just changed
        # - rewrite DD options based on new frame
        frame_idx = valid_frame_idx(frame)
        gt_points = GroundTruth.get(frame_idx, set())
        new_options = [f"x:{t[0]} y:{t[1]}" for t in gt_points]
        return new_options


    elif trigger_id == "update-gt-btn" and is_valid_gt_dd_value(dd_value):
        # Update current selected point with new one, based on x/y value
        # - update dict
        # - update DD options
        frame_idx = valid_frame_idx(frame)
        new_x = x_ipt if isinstance(x_ipt, int) else 0
        new_y = y_ipt if isinstance(y_ipt, int) else 0
        old_gt = convert_dd_value_to_gt(dd_value)
        new_options = dd_options if isinstance(dd_options, list) else []

        discard_ground_truth(frame_idx, old_gt)
        insert_ground_truth(frame_idx, new_x, new_y)

        with suppress(ValueError):
            new_options.remove(dd_value)
            new_options.append(f"x:{new_x} y:{new_y}")

        return new_options


    elif trigger_id == "remove-gt-btn" and is_valid_gt_dd_value(dd_value):
        # Remove current point:
        # - remove point from dict
        # - remove point from DD options
        old_gt = convert_dd_value_to_gt(dd_value)
        new_options = dd_options if isinstance(dd_options, list) else []
        frame_idx = valid_frame_idx(frame)

        discard_ground_truth(frame_idx, old_gt)
        with suppress(ValueError):
            new_options.remove(dd_value)
        
        return new_options

    else:
        # anything else: initial call for example, or a graph input with click_data=None
        # - just return options if any, otherwise empty list
        if isinstance(dd_options, list):
            return dd_options
        else:
            return []


@callback(
        Output("point-summary-row", "children"),
        Input("gt-point-dd", "options"),
)
def update_point_summary_row(gt_dd: list[str]) -> str:
    if not isinstance(gt_dd, list) or len(gt_dd) == 0:
        return ["No points placed yet"]
    return "\n".join(gt_dd)


@callback(
        Output("save-bt", "children"),
        Input("save-bt", "n_clicks"),
)
def save_ground_truth(n_clicks: int) -> str:
    if isinstance(n_clicks, int) and n_clicks >= 1:
        max_gt_frame_idx = max(list(GroundTruth.keys()))
        for f in range(max_gt_frame_idx+1):
            for p in GroundTruth.get(f, []):
                GTPoint(p[0], p[1])

        points = [[GTPoint(p[0], p[1]) for p in GroundTruth.get(f, [])] for f in range(max_gt_frame_idx+1)]

        gt = GTContainer(
            Src.path,
            "manual",
            points
        )
        gt.save_to(OutFp)
        return "Ground Truth successfully saved"
    else:
        return "Press to save your annotations"



###### Helpers ######

def valid_frame_idx(idx: Optional[int]) -> int:
    """Returns a valid frame index from the provided index"""
    if not isinstance(idx, int):
        return 0
    else:
        return max(0, min(idx, MovieData.get_max_frame_idx()))


def get_xy_from_click_data(click_data: dict) -> Tuple[int, int]:
    x = click_data["points"][0]["x"]
    y = click_data["points"][0]["y"]
    return x, y


def insert_ground_truth(frame: int, x: int, y: int):
    global GroundTruth
    gt_points = GroundTruth.get(frame)
    
    if gt_points is None:
        gt_points = {(x, y)}
    else:
        gt_points.add((x, y))

    GroundTruth[frame] = gt_points

def discard_ground_truth(frame: int, xy: Tuple[int, int]):
    global GroundTruth
    gt_points = GroundTruth.get(frame)
    if gt_points is None:
        return
    else:
        gt_points.discard(xy)
        GroundTruth[frame] = gt_points

def is_valid_gt_dd_value(value: Optional[str]) -> bool:
    # 5 because the smaller well-formatted value is 'x:0 y:0'
    return isinstance(value, str) and len(value) >= 7


def convert_dd_value_to_gt(value: str) -> Tuple[int, int]:
    # expected format is x:... y:...
    v_split = value.split(" ")
    x_str = v_split[0]
    y_str = v_split[1]
    x = int(x_str[2:], base=10)
    y = int(y_str[2:], base=10)
    return x, y

###### Entry ########
def start_app(src: Source, out_fp: str):
    global Src, OutFp, Data, MovieData
    Src = src
    OutFp = out_fp

    Data = src.load()
    MovieData = Movie(Data, Src.path)
    
    app.run(debug=True)
