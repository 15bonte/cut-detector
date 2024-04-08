from dash import callback, Input, Output, State, dcc, html, no_update
import dash_bootstrap_components as dbc
import plotly.express as px

from typing import Optional

def generate_slider_side_buttons(slider_id: str, marks: bool) -> dbc.Row:
    right_button = dbc.Button(
        id=f"{slider_id}-right-button",
        children=">", 
        outline=True, 
        color="primary", 
        className="me-1"
    ),
    left_button = dbc.Button(
        id=f"{slider_id}-left-button",
        children="<", 
        outline=True, 
        color="primary", 
        className="me-1"
    ),
    if marks:
        slider = dcc.Slider(
            id=slider_id,
            value=0,
            min=0,
            max=0,
            step=1,
        )
    else:
        slider = dcc.Slider(
            id=slider_id,
            value=0,
            min=0,
            max=0,
            step=1,
            marks=None,
            tooltip={"placement": "bottom", "always_visible": True}
        )
    return dbc.Row([
        dbc.Col(left_button, width=2),
        dbc.Col(slider, width=8),
        dbc.Col(right_button, width=2),
    ])


@callback(
    Output("detection-layer-slider", "value", allow_duplicate=True),
    Input("detection-layer-slider-right-button", "n_clicks"),
    State("detection-layer-slider", "value"),
    prevent_initial_call=True,
)
def update_layer_slider_left_button(n_clicks: int, slider_value: int):
    return slider_value + 1


@callback(
    Output("detection-layer-slider", "value", allow_duplicate=True),
    Input("detection-layer-slider-left-button", "n_clicks"),
    State("detection-layer-slider", "value"),
    prevent_initial_call=True,
)
def update_layer_slider_right_button(n_clicks: int, slider_value: int):
    return slider_value - 1


@callback(
    Output("detection-frame-slider", "value", allow_duplicate=True),
    Input("detection-frame-slider-right-button", "n_clicks"),
    State("detection-frame-slider", "value"),
    prevent_initial_call=True,
)
def update_frame_slider_left_button(n_clicks: int, slider_value: int):
    return slider_value + 1


@callback(
    Output("detection-frame-slider", "value", allow_duplicate=True),
    Input("detection-frame-slider-left-button", "n_clicks"),
    State("detection-frame-slider", "value"),
    prevent_initial_call=True,
)
def update_frame_slider_right_button(n_clicks: int, slider_value: int):
    return slider_value - 1


@callback(
    Output("detection-frame-slider", "value", allow_duplicate=True),
    Input("detection-frame-input", "value"),
    prevent_initial_call=True,
)
def update_frame_slider_from_input(value: Optional[int]):
    if isinstance(value, int):
        return value
    else:
        return no_update
