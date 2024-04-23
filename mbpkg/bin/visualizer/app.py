""" Main app file
"""

import contextlib
import os

import dash_mantine_components as dmc
from dash import Dash, html, dcc

from mbpkg.better_detector import Detector


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

def make_main_layout() -> list:
    from .components.v_tab_bar import make_vertical_tab_bar, DEFAULT_PANNEL
    from .components.pannel import make_pannel
    from .components.viewer import make_viewer
    from .components.viz_pannel import DEFAULT_SIG_VIZ_PARAM

    return dmc.Grid(
        columns=16,
        children=[
            # Globals signals
            dcc.Store(id="sig_open_pannel", data=DEFAULT_PANNEL),
            dcc.Store(id="sig_cur_frame", data=0),
            dcc.Store(id="sig_imp_file", data=None),
            dcc.Store(id="sig_det_param", data={}),
            dcc.Store(id="sig_cur_det", data=None),
            dcc.Store(id="sig_viz_param", data=DEFAULT_SIG_VIZ_PARAM),

            # The 3 pannels
            dmc.GridCol(make_vertical_tab_bar(), span=1),
            dmc.GridCol(make_pannel(), span=5),
            dmc.GridCol(make_viewer(), span=10),
        ]
    )



def start_app(mitosis_src_dirpaths: list[str], detectors: list[Detector]):
    from .components.shared import init_mitosis_src_dirpaths, init_detectors

    assert len(mitosis_src_dirpaths) > 0, "At least 1 movie src directory must be provided"
    init_mitosis_src_dirpaths(mitosis_src_dirpaths)

    assert len(detectors) > 0, "At least one detector must be provided"
    init_detectors(detectors)

    with react_env_wrapper():
        app = Dash(__name__, suppress_callback_exceptions=True)

        app.title = "Midbody Detection and Tracking Visualizer"

        app.layout = dmc.MantineProvider(
            forceColorScheme="dark",
            children = make_main_layout()
        )

        app.run(debug=True)