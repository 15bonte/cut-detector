"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from magicgui import magic_factory

from cut_detector.utils.tools import re_organize_channels
from cut_detector.widget_functions.mitosis_track_generation import perform_mitosis_track_generation

if TYPE_CHECKING:
    import napari


@magic_factory(
    call_button="Run Mitosis Track Generation",
    layout="vertical",
    xml_model_dir=dict(
        widget_type="FileEdit",
        label=".xml models directory: ",
    ),
    mitoses_save_dir=dict(
        widget_type="FileEdit",
        label="Mitoses saved path: ",
    ),
    tracks_save_dir=dict(
        widget_type="FileEdit",
        label="Tracks saved path: ",
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
    mitoses_path=dict(
        widget_type="FileEdit",
        label="Mitoses saved path: ",
    ),
)
def mid_body_detection(img_layer: "napari.layers.Image", mitoses_path: str):
    raw_video = re_organize_channels(img_layer.data)  # TXYC


@magic_factory(
    call_button="Run MT Cut Detection",
    layout="vertical",
    mitoses_path=dict(
        widget_type="FileEdit",
        label="Mitoses saved path: ",
    ),
)
def micro_tubules_cut_detection(img_layer: "napari.layers.Image", mitoses_path: str):
    raw_video = re_organize_channels(img_layer.data)  # TXYC
