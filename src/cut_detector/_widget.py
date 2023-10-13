"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from pathlib import Path
from typing import TYPE_CHECKING

from magicgui import magic_factory


from .utils.tools import re_organize_channels

from .widget_functions.tracking import perform_tracking
from .widget_functions.mid_body_detection import perform_mid_body_detection
from .widget_functions.mitosis_track_generation import perform_mitosis_track_generation
from .widget_functions.mt_cut_detection import perform_mt_cut_detection
from .widget_functions.save_results import perform_results_saving

if TYPE_CHECKING:
    import napari


@magic_factory(
    call_button="Run Segmentation and Tracking",
    layout="vertical",
    video_path=dict(
        widget_type="FileEdit",
        label="Video path",
    ),
    fiji_dir=dict(
        widget_type="FileEdit",
        label="Fiji path (likely named Fiji.app): ",
        mode="d",
    ),
    xml_model_dir=dict(
        widget_type="FileEdit",
        label=".xml models directory to save: ",
        mode="d",
    ),
    default_model_check_box=dict(
        widget_type="CheckBox", text="Use default segmentation model?", value=True
    ),
    segmentation_model=dict(
        widget_type="FileEdit",
        label="If not checked, cellpose segmentation model: ",
    ),
    fast_mode_check_box=dict(widget_type="CheckBox", text="Enable fast mode?", value=False),
)
def segmentation_tracking(
    video_path: str,
    fiji_dir: str,
    xml_model_dir: str,
    default_model_check_box: str,
    segmentation_model: str,
    fast_mode_check_box: bool,
):
    # Convert video path from windows to linux
    video_path = str(Path(video_path))
    segmentation_model = str(Path(segmentation_model))
    perform_tracking(
        video_path,
        fiji_dir,
        xml_model_dir,
        segmentation_model if not default_model_check_box else None,
        fast_mode_check_box,
    )
    print("\nSegmentation and tracking finished with success!")


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
    print("\nMitosis tracks generated with success!")


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
    print("\nMid-body detection finished with success!")


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
    print("\nMicro-tubules cut detection finished with success!")


@magic_factory(
    call_button="Save results",
    layout="vertical",
    exported_mitoses_dir=dict(
        widget_type="FileEdit",
        label="Saved .bin mitoses directory: ",
        mode="d",
    ),
    results_save_dir=dict(
        widget_type="FileEdit",
        label="Directory to save .bin mitoses: ",
        mode="d",
    ),
)
def results_saving(exported_mitoses_dir: str, results_save_dir: str):
    perform_results_saving(exported_mitoses_dir, save_dir=results_save_dir)
    print("\nResults saved with success!")
