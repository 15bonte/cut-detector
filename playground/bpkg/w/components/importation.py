from dash import Dash, callback, Input, Output, State, dcc, html, no_update
import dash_bootstrap_components as dbc

import os
import os.path
from typing import List
from ...data_loader import Movie

DEFAULT_SRC_DIR = "./src/cut_detector/data/mid_bodies_movies_test"

def render_blob_importation() -> List:
    src_dir = dbc.InputGroup(
        children=[
            dbc.InputGroupText(">>"),
            dbc.Input(
                id="importation-source-dir", 
                placeholder="Enter Source Directory",
                value=DEFAULT_SRC_DIR,
            )
        ],
        className="mb-3",
    ),
    src_choice = dbc.Select(
        id="importation-source-filename",
        options=[],
        value="",
    )
    src_kind = dcc.RadioItems(
        id="importation-source-kind",
        options=["4c", "3c"],
        value="4c"
    )
    imp_btn = dbc.Button(
        id="importation-savestate-btn",
        children="Confirm the Importation", 
        color="primary", 
        className="me-1"
    )
    imp_state = dbc.Alert(
        id="importation-state", 
        children="No source directory chosen", 
        color="danger"
    )
    l = [
        dbc.Row(src_dir),
        dbc.Row(src_choice),
        dbc.Row(src_kind),
        dbc.Row(imp_btn),
        dbc.Row(imp_state)
    ]
    return l

@callback(
    Output("importation-source-filename", "options"),
    Output("importation-state", "children", allow_duplicate=True),
    Output("importation-state", "color", allow_duplicate=True),
    Input("importation-source-dir", "value"),
    prevent_initial_call=True,
)
def update_src_choice_dd(source_dir: str):
    try:
        filenames = [f.name for f in os.scandir(source_dir) if f.is_file(follow_symlinks=False)]
        filenames.sort()
        return filenames, "Valid Source Directory, please select a file", "secondary"
    except FileNotFoundError: # invalid directory
        return [], f"Invalid Source Directory {source_dir}", "danger"
    
@callback(
    Output("importation-state", "children", allow_duplicate=True),
    Output("importation-state", "color", allow_duplicate=True),
    Input("importation-source-filename", "value"),
    State("importation-source-dir", "value"),
    prevent_initial_call=True,
)
def update_state_for_source_file_chosen(src_filename: str, src_dir: str):
    fp = os.path.join(src_dir, src_filename)
    if os.path.exists(fp):
        return "Valid Source File, Waiting for Confirmation", "info"
    else:
        return f"Invalid file '{fp}'", "danger"

@callback(
    Output("movie-info", "data"),
    Output("importation-state", "children", allow_duplicate=True),
    Output("importation-state", "color", allow_duplicate=True),
    Input("importation-savestate-btn", "n_clicks"),
    State("importation-source-kind", "value"),
    State("importation-source-dir", "value"),
    State("importation-source-filename", "value"),
    prevent_initial_call=True,
)
def update_movie_info(n_clicks: int, src_kind: str, src_dir: str, src_name: str):
    if len(src_dir) == 0 or len(src_name) == 0:
        return no_update, no_update, no_update
    
    fp = os.path.join(src_dir, src_name)
    if os.path.exists(fp):
        info = make_info(src_kind, fp)
        return info, "Importation Successful", "success"
    else:
        return no_update, f"Invalid File '{fp}'", "danger"

def make_info(kind: str, fp: str):
    if kind == "3c" or kind == "4c":
        return f"{kind} {fp}"
    else:
        raise RuntimeError(f"Invalid kind {kind}")