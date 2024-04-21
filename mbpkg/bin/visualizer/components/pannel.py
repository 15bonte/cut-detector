""" Drives which pannel to actually show
"""

from dash import callback, Input, Output, html, no_update
import dash_mantine_components as dmc

from .imp_pannel import make_imp_pannel
from .det_pannel import make_det_pannel

######## Constants ########
def make_trk_pannel() -> str:
    return "[tracking]"

def make_viz_pannel() -> str:
    return "[vizualisation]"

REQUEST_MAPPING = {
    "imp": make_imp_pannel,
    "det": make_det_pannel,
    "trk": make_trk_pannel,
    "viz": make_viz_pannel,
}
DEFAULT_PANNEL_FUNC = make_imp_pannel

######## Callbacks ########
@callback(
    Output("pannel_area", "children"),
    Input("sig_open_pannel", "data")
)
def update_pannel(request: str) -> list:
    if (isinstance(request, str) 
        and ((mk_func := REQUEST_MAPPING.get(request)) is not None)):
        return mk_func()
    
    else:
        return no_update

######## Layout ########
def make_pannel() -> dmc.Paper:
    return dmc.Paper(
        id="pannel_area",
        children=DEFAULT_PANNEL_FUNC(),
        style={"height": "100vh"},
        withBorder=True,
    )