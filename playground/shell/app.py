import contextlib
import os
import os.path
import pickle
from pathlib import Path
from typing import Tuple, Any

import numpy as np
from aicsimageio.aics_image import AICSImage
import dash_mantine_components as dmc
from dash import Dash, callback, Input, Output, State, ctx, no_update, dcc, html, ALL
from dash.exceptions import PreventUpdate
import plotly.express as px

from cut_detector.factories.mid_body_detection_factory import MidBodyDetectionFactory
from cut_detector.utils.mitosis_track import MitosisTrack

from app_config import EVAL_DATA_DIRS
from app_ext import AVAILABLE_DETECTORS

# LAYERS = [
#     "cells",
#     "sir",
#     "midbody",
# ]
LAYERS = {
    "cells": 2,
    "sir": 0,
    "midbody": 1,
    "mask": 3,
    "sigma_layer": -1
}
MB_FACTORY = MidBodyDetectionFactory()

_movie_data:    np.ndarray   = None # TYXC | C={sir:0 mb:1 cell:2 mask:3}
_mitosis_track: MitosisTrack = None
_id_dict: dict[int, str] = {}
_param_dict: dict[str, Any] = {}

@contextlib.contextmanager
def react_env_wrapper():
    env = os.environ
    try:
        env["REACT_VERSION"] = "18.2.0"
        print("react added")
        yield
    finally:
        del env["REACT_VERSION"]
        print("react removed")

def make_header_children() -> list:
    return dmc.Group([
        dmc.Select(
            id="dirname_sel",
            label="Directory",
            data=list(EVAL_DATA_DIRS.keys()),
            value=list(EVAL_DATA_DIRS.keys())[0],
            style={"width": "10vw"}
        ),
        dmc.Select(
            id="binfilter_sel",
            label="Filter",
            data=["All", "Valid GT/Div", "Wrong Detections"],
            value="All",
            style={"width": "20vw"},
        ),
        dmc.Select(
            id="filename_sel",
            label="File",
            data=[],
            searchable=True,
            style={"width": "50vw"},
        ),
    ])

@callback(
    Output("filename_sel", "data"),
    Input("dirname_sel", "value"),
    Input("binfilter_sel", "value")
)
def update_filename_sel(dirname: str, filter: str):
    dirname = check_ready_str(dirname)
    filter = check_ready_str(filter)
    with os.scandir(os.path.join(EVAL_DATA_DIRS[dirname], "mitoses")) as it:
        filenames = [e.name for e in it if e.is_file(follow_symlinks=False) and e.name.endswith(".bin")]
        filtered_filenames = []
        if filter == "Valid GT/Div":
            for binname in filenames:
                bin_path = binpath_from_binname_dirname(binname, dirname)
                with open(bin_path, "rb") as b:
                    track: MitosisTrack = pickle.load(b)
                    track.adapt_deprecated_attributes()
                    if track.gt_mid_body_spots is None or len(track.daughter_track_ids) != 1:
                        continue
                    else:
                        filtered_filenames.append(binname)
            filenames = filtered_filenames
        elif filter == "Wrong Detections":
            for binname in filenames:
                bin_path = binpath_from_binname_dirname(binname, dirname)
                with open(bin_path, "rb") as b:
                    track: MitosisTrack = pickle.load(b)
                    track.adapt_deprecated_attributes()
                    if track.gt_mid_body_spots is None or len(track.daughter_track_ids) != 1:
                        continue
                    else:
                        is_correct, _, _ = track.evaluate_mid_body_detection()
                        if not is_correct:
                            filtered_filenames.append(binname)
            filenames = filtered_filenames
        elif filter != "All":
            raise RuntimeError(f"Unsupported filter value {filter}")

    filenames.sort()
    return filenames

@callback(
    Output("movie_loaded", "data"),
    Input("filename_sel", "value"),
    State("dirname_sel", "value"),
)
def update_movie_and_track(bin_filename: str, dirname: str) -> str:
    global _movie_data
    global _mitosis_track

    bin_filename = check_ready_str(bin_filename)
    dirname      = check_ready_str(dirname)

    tiff_path = tiffpath_from_binname_dirname(bin_filename, dirname)
    _movie_data = read_tiff(tiff_path)

    bin_path = Path(EVAL_DATA_DIRS[dirname]) / Path("mitoses") / Path(bin_filename)
    with open(bin_path, "rb") as f:
        _mitosis_track = pickle.load(f)

    return f"{dirname}/{bin_filename}"

def make_navbar_children() -> list:
    return dmc.ScrollArea(dmc.Stack([
        dmc.Card([
            dmc.CardSection("Rendering"),
            dmc.Select(
                id="layer_sel",
                # data=LAYERS,
                # value=LAYERS[0],
                data=list(LAYERS.keys()),
                value=list(LAYERS.keys())[0],
            ),
            dmc.Switch(
                id="spots_switch",
                label="Show GT and test spots",
                checked=True,
            ),
            dmc.Switch(
                id="mask_switch",
                label="Show approximated mask",
                checked=False,
            ),
            dmc.Switch(
                id="detection_switch",
                label="Run detection on current frame",
                checked=False,
            ),
        ]),
        dmc.Card([
            dmc.CardSection("Live detection"),
            dmc.Select(
                id="detector_sel",
                data=list(AVAILABLE_DETECTORS.keys()),
                value=list(AVAILABLE_DETECTORS.keys())[0],
            ),
            html.Div(id="detector_param_area")
        ]),
        dmc.Card([
            dmc.CardSection("Graph Control"),
            dmc.NumberInput(
                id="frame_input",
                label="Frame",
                value=0,
                min=0,
                max=0,
                rightSection=dmc.Text(id="frame_input_max_text", children="/0"),
            ),
            dmc.Group([
                dmc.Button(id="previous_btn", children="<", color="gray", variant="outline"),
                dmc.Button(id="next_btn", children=">", color="gray", variant="outline")
            ], grow=True),
        ]),
        dmc.Card([
            dmc.CardSection("Spots"),
            dmc.Divider(label="Ground Truth Spots:", labelPosition="center"),
            dmc.ScrollArea(html.Div(id="gt_spot_list"), h="30vh"),
            dmc.Divider(label="Found Spots:", labelPosition="center"),
            dmc.ScrollArea(html.Div(id="test_spot_list"), h="30vh"),
        ])
    ]))

@callback(
    Output("detector_param_area", "children"),
    Input("detector_sel", "value")
)
def update_detector_param_area(detector_name: str) -> list:
    global _id_dict, _param_dict
    _id_dict = {}
    _param_dict = {}

    detector_name = check_ready_str(detector_name)
    ext = AVAILABLE_DETECTORS[detector_name]

    l = ext.generate_and_bind_widgets(_id_dict)
    ext.initialize_param_dict(_param_dict)

    return l

@callback(
    Output("param_dict_updated", "data"),
    Input({"type": "detector_widget", "id": ALL}, "value"),
    State("param_dict_updated", "data")
)
def update_param_dict(values: list[float], n_update: int):
    global _param_dict

    for id, v in enumerate(values):
        _param_dict[_id_dict[id]] = v

    return n_update+1 if isinstance(n_update, int) else 0

@callback(
    Output("frame_input", "value"),
    Input("previous_btn", "n_clicks"),
    Input("next_btn", "n_clicks"),
    Input("movie_loaded", "data"),
    State("frame_input", "value")
)
def update_frame_input(p_btn: int, n_btn: int, movie_loaded: str, cur_value: int) -> int:
    id = ctx.triggered_id
    if id == "previous_btn":
        check_ready_int(p_btn)
        return max(0, cur_value-1)
    
    elif id == "next_btn":
        check_ready_int(n_btn)
        return min(cur_value+1, _movie_data.shape[0]-1)
    
    elif id == "movie_loaded":
        check_ready_str(movie_loaded)
        return 0
    
    else:
        return no_update


@callback(
    Output("frame_input", "max"),
    Output("frame_input_max_text", "children"),
    Input("movie_loaded", "data")
)
def update_max_frame(movie_loaded: str) -> Tuple[int, str]:
    _ = check_ready_str(movie_loaded)
    t_max_index = _movie_data.shape[0] -1
    return t_max_index, f"/{t_max_index}"

@callback(
    Output("graph", "figure"),
    Input("movie_loaded", "data"),
    Input("frame_input", "value"),
    Input("layer_sel", "value"),
    Input("spots_switch", "checked"),
    Input("mask_switch", "checked"),
    Input("detection_switch", "checked"),
    State("detector_sel", "value"),
    Input("param_dict_updated", "data")
)
def update_graph(
        movie_loaded: str, 
        frame: int, 
        layer: str,
        show_spots: bool,
        show_mask: bool,
        run_detection: bool,
        detector_name: str,
        param_dict_updated: int
        ) -> dict:
    check_ready_str(movie_loaded)
    check_ready_int(param_dict_updated)
    frame = check_ready_int(frame)
    layer = check_ready_str(layer)
    show_spots = check_ready_bool(show_spots)
    show_mask = check_ready_bool(show_mask)
    run_detection = check_ready_bool(run_detection)
    detector_name = check_ready_str(detector_name)

    if show_mask and (layer_index := LAYERS[layer]) >= 0:
        fig = px.imshow(
            _movie_data[frame, :, :, layer_index] * _movie_data[frame, :, :, 3],
            template="plotly_dark"
        )
    elif (layer_index := LAYERS[layer]) >= 0:
        fig = px.imshow(
            _movie_data[frame, :, :, layer_index],
            template="plotly_dark"
        )
    else:
        raise RuntimeError("For now, special negative layers are not supported")

    if show_spots:
        if (_mitosis_track.gt_mid_body_spots is not None 
            and ((spot := _mitosis_track.gt_mid_body_spots.get(_mitosis_track.min_frame + frame)) is not None)
            ):
            x0 = spot.x - 8
            x1 = spot.x + 8
            y0 = spot.y - 8
            y1 = spot.y + 8
            fig.add_shape(
                type="circle",
                xref="x", yref="y",
                x0=x0, y0=y0, x1=x1, y1=y1,
                line=dict(
                    color="DarkGreen",
                    width=4,
                )
            )

        if (spot := _mitosis_track.mid_body_spots.get(_mitosis_track.min_frame + frame)) is not None:
            x0 = spot.x - 10
            x1 = spot.x + 10
            y0 = spot.y - 10
            y1 = spot.y + 10
            fig.add_shape(
                type="circle",
                xref="x", yref="y",
                x0=x0, y0=y0, x1=x1, y1=y1,
                line=dict(
                    color="DarkBlue",
                    width=4,
                )
            )
    
    if run_detection:
        ext = AVAILABLE_DETECTORS[detector_name]
        spots = MB_FACTORY._spot_detection(
            image=_movie_data[frame,:,:,:],
            mask=_movie_data[frame,:,:,3],
            mid_body_channel=1,
            sir_channel=0,
            mode=ext.make_detector(_param_dict),
            frame=frame,
            log_blob_spot=False,
            mitosis_track=_mitosis_track
        )
        for spot in spots:
            x0 = spot.x - 12
            x1 = spot.x + 12
            y0 = spot.y - 12
            y1 = spot.y + 12
            fig.add_shape(
                type="circle",
                xref="x", yref="y",
                x0=x0, y0=y0, x1=x1, y1=y1,
                line=dict(
                    color="FireBrick",
                    width=4,
                )
            )

    return fig

@callback(
    Output("gt_spot_list", "children"),
    Output("test_spot_list", "children"),
    Input("movie_loaded", "data")
)
def update_spot_lists(movie_loaded: str):
    check_ready_str(movie_loaded)
    if _mitosis_track.gt_mid_body_spots is None:
        gt_spots = []
    else:
        gt_spots = [dmc.Text(f"f:{f-_mitosis_track.min_frame} x:{s.x} y:{s.y}") for f, s in _mitosis_track.gt_mid_body_spots.items()]
    test_spots = [dmc.Text(f"f:{f-_mitosis_track.min_frame} x:{s.x} y:{s.y}") for f, s in _mitosis_track.mid_body_spots.items()]
    return gt_spots, test_spots

def check_ready_str(s: str) -> str:
    if isinstance(s, str) and len(s) > 0:
        return s
    else:
        raise PreventUpdate
    
def check_ready_int(i: int) -> int:
    if isinstance(i, int):
        return i
    else:
        raise PreventUpdate
    
def check_ready_bool(b: bool) -> bool:
    if isinstance(b, bool):
        return b
    else:
        raise PreventUpdate
    
def tiffpath_from_binname_dirname(binname: str, dirname: str) -> Path:
    tiff_dir = Path(EVAL_DATA_DIRS[dirname]) / Path("movies")
    tiffname = Path(f"{Path(binname).stem}.tiff")
    return tiff_dir / tiffname

def binpath_from_binname_dirname(bin_filename: str, dirname: str) -> Path:
    bin_filename = check_ready_str(bin_filename)
    bin_path = Path(EVAL_DATA_DIRS[dirname]) / Path("mitoses") / Path(bin_filename)
    return bin_path

def read_tiff(path: Path) -> np.ndarray:
    # TCZYX -> TCYX -> TYXC
    return AICSImage(path).data.squeeze().transpose(0, 2, 3, 1)

def start_app():
    with react_env_wrapper():
        app = Dash(__name__)
        app.title = "Midbody/Mitosis Shell"

        app.layout = dmc.MantineProvider(
            forceColorScheme="dark",
            children=dmc.AppShell(
                [
                    dmc.AppShellHeader(make_header_children()),
                    dmc.AppShellNavbar(make_navbar_children()),
                    dmc.AppShellMain(children=[
                        dcc.Store("movie_loaded", data=None),
                        dcc.Store("param_dict_updated", data=None),

                        dcc.Graph(
                            id="graph",
                            figure={},
                            style={"height": "77vh"}
                        )
                    ]),
                    dmc.AppShellFooter("Footer")
                ],
                header={"height": "8vh"},
                navbar={
                    "width": "20vw",
                    "breakpoint": "sm",
                    "collapsed": {"mobile": True},
                },
                footer={"height": "8vh"},
                padding="xl",
            )
        )

        app.run(debug=True)