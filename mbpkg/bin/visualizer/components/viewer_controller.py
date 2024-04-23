from typing import Tuple

from dash import callback, Input, Output, State, ctx, no_update
import dash_mantine_components as dmc
import dash_bootstrap_components as dbc

from mbpkg.movie_loading import Movie


######## Callbacks ########
@callback(
    Output("sig_cur_frame", "data"),
    Input("vctrl_frame_input", "value")
)
def update_sig_cur_frame(f: int) -> int:
    if isinstance(f, int):
        return f
    else:
        return 0

@callback(
    Output("vctrl_frame_input", "value"),
    Input("vctrl_pbtn", "n_clicks"),
    Input("vctrl_nbtn", "n_clicks"),
    State("vctrl_frame_input", "value")
)
def update_frame_input(p_btn: int, r_btn: int, cur_frame: int) -> int:
    from .shared import _movie

    id = ctx.triggered_id
    new_frame = cur_frame if isinstance(cur_frame, int) else 0
    movie_max = _movie.get_max_frame_idx() if isinstance(_movie, Movie) else 0

    if id == "vctrl_pbtn":
        new_frame -= 1
        return max(0, min(new_frame, movie_max))
    elif id  == "vctrl_nbtn":
        new_frame += 1
        return max(0, min(new_frame, movie_max))
    else:
        return no_update


@callback(
    Output("vctrl_frame_input", "max"),
    Output("vctr_frame_max_text", "children"),
    Input("sig_imp_file", "data")
)
def update_max_frame(sig_imp: str) -> int:
    from .shared import _movie
    if isinstance(_movie, Movie) and isinstance(sig_imp, str) and len(sig_imp) > 0:
        max_frame = _movie.get_max_frame_idx()
        return max_frame, f"/{max_frame}"
    else:
        return 0, "/0"


######## Layout ########

def make_viewer_controller() -> dmc.Group:
    return dmc.Grid([
        dmc.GridCol(dmc.Button(id="vctrl_pbtn", children="<"), span=1),
        dmc.GridCol(dmc.NumberInput(
            id="vctrl_frame_input",
            value=0,
            leftSection=dmc.Text("Frame:"),
            leftSectionWidth=75,
            rightSection=dmc.Text(id="vctr_frame_max_text", children="/00"),
            rightSectionWidth=100,
            min=0,
            max=0,
        ), span=10),
        dmc.GridCol(dmc.Button(id="vctrl_nbtn", children=">"), span=1),
    ], grow=True)

