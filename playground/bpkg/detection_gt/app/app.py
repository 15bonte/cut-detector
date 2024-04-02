from dash import Dash, html
from typing import Optional

SrcFp: Optional[str] = None
OutFp: Optional[str] = None

app = Dash(__name__)

app.layout = html.Div([
    html.Div(children='Hello World')
])

def start_app(src_fp: str, out_fp: str):
    global SrcFp, OutFp
    SrcFp = src_fp
    OutFp = out_fp
    
    app.run()
