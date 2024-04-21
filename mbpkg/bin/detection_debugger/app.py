""" A debugging and visualization app written in Dash.
IDs can be copy-pasted from ids.txt
"""

from dash import Dash, html, dash_table, dcc, callback, Output, Input
import dash_bootstrap_components as dbc

from mbpkg.detector import Detector

from .vis_area import gen_visualization_area
from .shared import *


def run_app(source_dirs: list[str], detectors: list[Detector]):
    if len(source_dirs) == 0:
        raise RuntimeError("source_dirs must contain at least 1 directory path")
    if len(detectors) == 0:
        raise RuntimeError("detectors must contain at least 1 detector")

    app = Dash(
        __name__, 
        external_stylesheets=[dbc.themes.BOOTSTRAP], # since we are using the boostrape cpmts
        suppress_callback_exceptions=True # IDs in tabs that are not present at the beginning
    )
    app.title = "Detection Debugger and Visualizer"

    app.layout = dbc.Container([
        dcc.Store(id="sig_import_movie_in",      data=None),
        dcc.Store(id="sig_import_movie_out",     data=None),

        gen_visualization_area(source_dirs, detectors)
    ], fluid=True, style={"height": "100vh"})

    app.run(debug=True)

