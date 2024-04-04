from dash import callback, Input, Output, State, dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px

from typing import List, Dict
from importation import Movie

LocalMovie: Movie = None

def render_blob_tracking() -> List:
    l = [
        html.P("WIP")
    ]
    return l