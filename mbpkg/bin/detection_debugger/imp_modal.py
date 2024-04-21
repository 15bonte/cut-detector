import os
import os.path
import json
from pathlib import Path

from mbpkg.movie_loading import MovieFmt

from dash import dcc, callback, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc

################## Callbacks ##################
@callback(
    Output("imp_filename_dd", "options"),
    Input("imp_dir_dd", "value")
)
def update_src_dd(dir_dd: str) -> list[str]:
    if not isinstance(dir_dd, str) and len(dir_dd) > 0:
        return no_update
    
    return list_movie_files(dir_dd)


@callback(
    Output("imp_load_btn", "children"),
    Input("imp_filename_dd", "value"),
    Input("sig_import_movie_out", "data"),
    State("imp_fmt_dd", "value")
)
def update_button_message(
        filename: str, 
        out_data: str,
        fmt: str
        ) -> str:
    if ctx.triggered_id == "imp_filename_dd" and isinstance(filename, str):
        fmt_str = fmt if isinstance(fmt, str) else "/"
        return f"Press to import [{filename} | {fmt_str}]"
    
    elif ctx.triggered_id == "sig_import_movie_out" \
            and isinstance(out_data, str) \
            and len(out_data) > 0:
        d = json.loads(out_data)
        p = Path(d["path"])
        fmt = d["fmt"]
        return f"File [{p.name} | {fmt}] successfully imported"
    
    else:
        return no_update


@callback(
    Output("sig_import_movie_in", "data"),
    Input("imp_load_btn", "n_clicks"),
    State("imp_filename_dd", "value"),
    State("imp_dir_dd", "value"),
    State("imp_fmt_dd", "value")
)
def press_import_button(
        n_clicks: int, 
        filename: str, 
        dirpath: str,
        fmt: str):
    
    for v in [filename, dirpath, fmt]:
        if not isinstance(v, str) or len(v) == 0:
            return None

    return json.dumps({
        "path": os.path.join(dirpath, filename),
        "fmt": fmt
    })

################## Helpers ##################

def list_movie_files(dirpath: str) -> list[str]:
    with os.scandir(dirpath) as it:
        src_names = [
            e.name
            for e in it if e.is_file(follow_symlinks=False) \
                and e.name.endswith((".tif", ".tiff"))
        ]
        src_names.sort()
    return src_names

################## Layout ##################

def gen_importation_modal_body(dir_dd_options: list[str]) -> dbc.Container:
    dir_path = dir_dd_options[0]
    src_names = list(dir_path)

    return dbc.Container([
        dbc.Row(dcc.Dropdown(
            id="imp_dir_dd",
            options=dir_dd_options,
            value=dir_path
        )),
        dbc.Row(dcc.Dropdown(
            id="imp_filename_dd",
            options=src_names
        )),
        dbc.Row(dcc.Dropdown(
            id="imp_fmt_dd",
            options=MovieFmt.available_fmt_strs(),
            value=MovieFmt.available_fmt_strs()[0],
        )),
        dbc.Row(dbc.Button(
            id="imp_load_btn",
            children="Press to Start the Import"
        ))
    ])