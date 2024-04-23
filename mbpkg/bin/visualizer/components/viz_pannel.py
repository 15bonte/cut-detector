from typing import Literal, Any, get_args

from dash import html, callback, Input, Output, State, no_update, ALL
import dash_mantine_components as dmc

######## Local Globals ########
viz_param_idx_to_arg: dict[int, str] = {}

######## Constants ########
LAYER_LIT = Literal[
    "cells",
    "midbody",
    "sir",
    "sigma_layer"
]
LAYERS = list(get_args(LAYER_LIT))

DEFAULT_SIG_VIZ_PARAM: dict[str, Any] = {
    "layer": "cells"
}

DEFAULT_SIGMA = 0.01

######## Callbacks ########
@callback(
    Output("viz_parameters", "children"),
    Input("viz_layer", "value")
)
def update_viz_parameters(layer: str) -> list:
    global viz_param_idx_to_arg

    if isinstance(layer, str) or len(layer) > 0:
        if layer in ["cells", "midbody", "sir"]:
            viz_param_idx_to_arg = {}
            return f"No additional parameter for {layer} layer"
        
        elif layer == "sigma_layer":
            viz_param_idx_to_arg[0] = "viz_sigma"
            return dmc.NumberInput(
                id={"type": "viz_param", "index": 0},
                label="Sigma",
                description="Choose sigma value to show for layer",
                value=DEFAULT_SIGMA
            )
        else:
            viz_param_idx_to_arg = {}
            raise VizError(f"Unknown layer {layer}")
    else:
        return no_update
    
@callback(
    Output("sig_viz_param", "data"),
    Input("viz_layer", "value"),
    Input({"type": "viz_param", "index": ALL}, "value"),
)
def update_sig_viz_param(layer: str, viz_param: list[float]):
    if isinstance(layer, str) and len(layer) > 0:
        d = {
            viz_param_idx_to_arg[idx]: v
            for idx, v in enumerate(viz_param)
        }
        d["layer"] = layer
        return d
    else:
        return no_update

######## Helpers ########
class VizError(Exception):
    pass

######## Layout ########
def make_viz_pannel() -> dmc.Stack:
    return dmc.Stack([
        dmc.Card([
            dmc.CardSection("Debug settings"),
            dmc.Select(
                id="viz_layer",
                data=LAYERS,
                value=LAYERS[0]
            ),
            html.Div(id="viz_parameters", children="[parameters here]")
        ]),
        dmc.Card([
            dmc.CardSection("Rendering settings"),
            dmc.Text("[rendering settings]")
        ])
    ], align="stretch", gap="md")

