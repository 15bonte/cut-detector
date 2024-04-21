""" Sharded globals
"""
import json

from dash import callback, Input, Output

from mbpkg.movie_loading import Movie, MovieFmt, Source


######### *Shared* Global Variables #########
_movie: Movie = None


######### Callbacks #########
@callback(
    Output("sig_import_movie_out", "data"),
    Input("sig_import_movie_in", "data")
)
def load_movie(in_data: str):
    global _movie

    if not isinstance(in_data, str) or len(in_data) == 0:
        return None
    
    d = json.loads(in_data)
    fmt  = d["fmt"]
    path = d["path"]

    _movie = Source(path, MovieFmt(fmt)).load_movie()
    print(f"Loaded mitosis movie [{path}] | {fmt}")

    return in_data