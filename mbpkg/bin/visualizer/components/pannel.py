""" Drives which pannel to actually show
"""

from dash import callback, Input, Output, html, no_update
import dash_mantine_components as dmc

from .imp_pannel import make_imp_pannel
from .det_pannel_2 import make_det_pannel
from .viz_pannel import make_viz_pannel
from .trk_pannel import make_trk_pannel

######## Constants ########
def make_all_pannel() -> str:
    cat_layout = [
        make_imp_pannel(),
        make_det_pannel(),
        make_trk_pannel(),
        make_viz_pannel(),
    ]
    return dmc.ScrollArea(dmc.Stack(
        children=cat_layout,
        align="stretch", gap="md",
        w="30vw",
    ))

REQUEST_MAPPING = {
    "imp": make_imp_pannel,
    "det": make_det_pannel,
    "trk": make_trk_pannel,
    "viz": make_viz_pannel,
    "all": make_all_pannel,
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