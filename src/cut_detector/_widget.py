"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from magicgui import magic_factory

from .utils.tools import re_organize_channels

from .widget_functions.mid_body_detection import perform_mid_body_detection
from .widget_functions.mitosis_track_generation import perform_mitosis_track_generation
from .widget_functions.mt_cut_detection import perform_mt_cut_detection

if TYPE_CHECKING:
    import napari


@magic_factory(
    call_button="Run Mitosis Track Generation",
    layout="vertical",
    xml_model_dir=dict(
        widget_type="FileEdit",
        label=".xml models directory: ",
        mode="d",
    ),
    mitoses_save_dir=dict(
        widget_type="FileEdit",
        label="Directory to save .bin mitoses: ",
        mode="d",
    ),
    tracks_save_dir=dict(
        widget_type="FileEdit",
        label="Directory to save .bin tracks: ",
        mode="d",
    ),
)
def mitosis_track_generation(
    img_layer: "napari.layers.Image",
    xml_model_dir: str,
    mitoses_save_dir: str,
    tracks_save_dir: str,
):
    raw_video = re_organize_channels(img_layer.data)  # TXYC
    perform_mitosis_track_generation(
        raw_video, img_layer.name, xml_model_dir, mitoses_save_dir, tracks_save_dir
    )


@magic_factory(
    call_button="Run Mid-body Detection",
    layout="vertical",
    exported_mitoses_dir=dict(
        widget_type="FileEdit",
        label="Saved .bin mitoses directory: ",
        mode="d",
    ),
    exported_tracks_dir=dict(
        widget_type="FileEdit",
        label="Saved .bin tracks directory: ",
        mode="d",
    ),
    save_check_box=dict(widget_type="CheckBox", text="Save cell divisions movies?", value=False),
    save_dir=dict(
        widget_type="FileEdit",
        label="If checked, directory to save division movies: ",
        mode="d",
    ),
)
def mid_body_detection(
    img_layer: "napari.layers.Image",
    exported_mitoses_dir: str,
    exported_tracks_dir: str,
    save_check_box: bool,
    save_dir: str,
):
    raw_video = re_organize_channels(img_layer.data)  # TXYC
    perform_mid_body_detection(
        raw_video,
        img_layer.name,
        exported_mitoses_dir,
        exported_tracks_dir,
        save_dir if save_check_box else None,
    )


@magic_factory(
    call_button="Run MT Cut Detection",
    layout="vertical",
    exported_mitoses_dir=dict(
        widget_type="FileEdit",
        label="Saved .bin mitoses directory: ",
        mode="d",
    ),
)
def micro_tubules_cut_detection(img_layer: "napari.layers.Image", exported_mitoses_dir: str):
    raw_video = re_organize_channels(img_layer.data)  # TXYC
    perform_mt_cut_detection(raw_video, img_layer.name, exported_mitoses_dir)
