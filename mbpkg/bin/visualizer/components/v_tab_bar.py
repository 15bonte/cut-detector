from dash import callback, Input, Output, ctx, no_update
import dash_mantine_components as dmc
from dash_iconify import DashIconify

######## Constants ########
ID_PANNEL_MAPPING = {
    "sb_imp_btn": "imp",
    "sb_det_btn": "det",
    "sb_trk_btn": "trk",
    "sb_viz_btn": "viz"
}
DEFAULT_PANNEL = ID_PANNEL_MAPPING["sb_imp_btn"]



######## Callbacks ########
@callback(
    Output("sig_open_pannel", "data"),
    Input("sb_imp_btn", "n_clicks"),
    Input("sb_det_btn", "n_clicks"),
    Input("sb_trk_btn", "n_clicks"),
    Input("sb_viz_btn", "n_clicks"),
)
def update_pannel(imp: int, det: int, trk: int, viz: int) -> str:
    if (s := ID_PANNEL_MAPPING.get(ctx.triggered_id)) is not None:
        return s
    else:
        return no_update


######## Layout ########
def make_vertical_tab_bar() -> dmc.Paper:
    return dmc.Paper(dmc.Stack([
        dmc.ActionIcon(
            id="sb_imp_btn",
            children=DashIconify(icon="bi:filetype-tiff"),
            color="blue"
        ),
        dmc.ActionIcon(
            id="sb_det_btn",
            children=DashIconify(icon="ph:magnifying-glass-duotone"),
            color="blue"
        ),
        dmc.ActionIcon(
            id="sb_trk_btn",
            children=DashIconify(icon="mdi:target"),
            color="blue"
        ),
        dmc.ActionIcon(
            id="sb_viz_btn",
            children=DashIconify(icon="healthicons:blood-cells"),
            color="blue"
        ),
    ], align="center", gap="xl", justify="center", style={"height": "100vh"}), withBorder=True)