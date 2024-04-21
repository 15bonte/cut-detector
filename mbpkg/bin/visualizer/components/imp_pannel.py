""" Import pannel is where one can import the files to use in the software
"""

from os import scandir

from dash import callback, Input, Output, State, no_update, ctx
import dash_mantine_components as dmc
from dash_iconify import DashIconify

from mbpkg.movie_loading import MovieFmt

from .shared import sig_load_movie

######## Callbacks ########
@callback(
    Output("imp_file_sel", "value"),
    Output("imp_file_sel", "data"),
    Input("imp_dir_sel", "value"),
)
def update_filename_select(dirpath: str) -> str:
    if isinstance(dirpath, str) and len(dirpath) > 0:
        movies = list_movie_filenames(dirpath)
        return movies[0], movies
    else:
        return no_update, no_update
    

@callback(
    Output("imp_btn", "children"),
    Input("imp_file_sel", "value"),
    Input("imp_fmt_sel", "value"),
    State("imp_dir_sel", "value"),

    Input("sig_imp_file", "data"),
)
def update_imp_button(filename: str, fmt: str, dirpath: str, sig_imp: str):
    id = ctx.triggered_id

    if id in ["imp_fmt_sel", "imp_file_sel"]:
        for s in [filename, fmt, dirpath]:
            if not isinstance(s, str) or len(s) == 0:
                return no_update

        return f"Press to import {filename} | {fmt}"
    
    elif id == "sig_imp_file" and isinstance(sig_imp, str):
        return "Mitosis movie imported successfully"
    
    else:
        return no_update


@callback(
    Output("sig_imp_file", "data"),
    Input("imp_btn", "n_clicks"),
    State("imp_dir_sel", "value"),
    State("imp_file_sel", "value"),
    State("imp_fmt_sel", "value"),
)
def send_import_signal(btn: int, dirpath: str, filename: str, fmt: str) -> str:
    if isinstance(btn, int) and btn > 0:
        for s in [dirpath, filename, fmt]:
            if not isinstance(s, str) or len(s) == 0:
                return no_update 

        return sig_load_movie(dirpath, filename, fmt)
    
    else:
        return no_update

######## Helpers ########
def list_movie_filenames(dirpath: str) -> list[str]:
    with scandir(dirpath) as it:
        l = [
            entry.name 
            for entry in it if (entry.is_file(follow_symlinks=False) 
                and entry.path.endswith((".tiff", ".tif")))
        ]
        l.sort()
        return l
    


######## Layout ########
def make_imp_pannel() -> list:
    from .shared import _mitosis_src_dirpaths

    default_dir = _mitosis_src_dirpaths[0]
    default_filenames = list_movie_filenames(default_dir)
    default_filename = default_filenames[0]

    return dmc.Stack([
        dmc.Select(
            id="imp_dir_sel",
            label="Choose a source directory", 
            data=_mitosis_src_dirpaths,
            value=default_dir,
            searchable=True,
            nothingFoundMessage="No directory match",
        ),
        dmc.Select(
            id="imp_file_sel",
            label="Choose a file", 
            data=default_filenames,
            value=default_filename,
            searchable=True,
            nothingFoundMessage="No file match",
        ),
        dmc.Select(
            id="imp_fmt_sel",
            label="Pick the right format", 
            data=MovieFmt.available_fmt_strs(),
            value=MovieFmt.available_fmt_strs()[0],
        ),
        dmc.Button(
            id="imp_btn",
            children="Load chosen file",
            leftSection=DashIconify(icon="uiw:upload"),
        ),
    ], align="stretch", gap="md")