from dash import Dash, callback, Input, Output, State, dcc, html
import dash_bootstrap_components as dbc

from .components.importation import render_blob_importation
from .components.blob_detection import render_blob_detection, release_detection_movie
from .components.blob_tracking import render_blob_tracking

app = Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP], # avoid dark theme for dcc
    suppress_callback_exceptions=True, # for tab layout
)

app.title = "Visualizer V2"

app.layout = dbc.Container(children=[
    dcc.Store(id="movie-info", data=""),
    html.H1("Blob Detection and Tracking Visualizer V2"),
    dcc.Tabs(id="workspace-tabs", value="ws-tab-importation", children=[
        dcc.Tab(label="Importation", value="ws-tab-importation"),
        dcc.Tab(label="Blob Detection", value="ws-tab-detection"),
        dcc.Tab(label="Blob Tracking", value="ws-tab-tracking"),
    ]),
    html.Div(id="workspace-tabs-content"),
])


@callback(
    Output("workspace-tabs-content", "children"),
    Input("workspace-tabs", "value")
)
def update_workspace_tab_content(ws_tab_choice):
    if ws_tab_choice == "ws-tab-importation":
        release_detection_movie()
        return render_blob_importation()
    elif ws_tab_choice == "ws-tab-detection":
        return render_blob_detection()
    elif ws_tab_choice == "ws-tab-tracking":
        release_detection_movie()
        return render_blob_tracking()
    else:
        return dbc.Alert(f"Tab value {ws_tab_choice} not supported", color="danger"),


def run_app():
    app.run(debug=True)