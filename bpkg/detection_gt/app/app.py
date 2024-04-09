from dash import Dash, html
from typing import Optional

from data_loading import Source

SrcFp: Optional[Source] = None
OutFp: Optional[str] = None

app = Dash(__name__)

app.layout = html.Div([
    html.Div(children='Hello World')
])

def start_app(src_fp: Source, out_fp: str):
    global SrcFp, OutFp
    SrcFp = src_fp
    OutFp = out_fp
    
    app.run(debug=True)
