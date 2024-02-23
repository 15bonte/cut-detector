import os
from pathlib import Path
from typing import Optional

from magicgui import magic_factory
import tempfile


from .utils.tools import re_organize_channels

from .widget_functions.tracking import perform_tracking
from .widget_functions.mid_body_detection import perform_mid_body_detection
from .widget_functions.mitosis_track_generation import (
    perform_mitosis_track_generation,
)
from .widget_functions.mt_cut_detection import perform_mt_cut_detection
from .widget_functions.save_results import perform_results_saving


@magic_factory(
    call_button="Run Whole Process",
    layout="vertical",
    fiji_dir=dict(
        widget_type="FileEdit",
        label="Fiji path (likely named Fiji.app): ",
        mode="d",
    ),
    default_model_check_box=dict(
        widget_type="CheckBox",
        text="Use default segmentation model?",
        value=True,
    ),
    segmentation_model=dict(
        widget_type="FileEdit",
        label="If not checked, cellpose segmentation model: ",
    ),
    fast_mode_check_box=dict(
        widget_type="CheckBox", text="Enable fast mode?", value=False
    ),
    save_check_box=dict(
        widget_type="CheckBox", text="Save cell divisions movies?", value=False
    ),
    movies_save_dir=dict(
        widget_type="FileEdit",
        label="If checked, directory to save division movies: ",
        mode="d",
    ),
    results_save_dir=dict(
        widget_type="FileEdit",
        label="Directory to save results: ",
        mode="d",
    ),
)
def whole_process(
    img_layer: "napari.layers.Image",
    img_viewer: "napari.Viewer",
    fiji_dir: str,
    default_model_check_box: str,
    segmentation_model: str,
    fast_mode_check_box: bool,
    save_check_box: bool,
    movies_save_dir: str,
    results_save_dir: str,
):
    # Make sure we are working with the same image (same name)
    video_path = None
    for layer in img_viewer.layers:
        local_video_path = layer.source.path
        if img_layer.name == os.path.basename(local_video_path).split(".")[0]:
            # This is the same image, we keep the link
            video_path = local_video_path
            break
    assert video_path is not None

    # Create temporary folders
    xml_model_dir_instance = tempfile.TemporaryDirectory()
    xml_model_dir = xml_model_dir_instance.name
    exported_mitoses_dir_instance = tempfile.TemporaryDirectory()
    exported_mitoses_dir = exported_mitoses_dir_instance.name
    exported_tracks_dir_instance = tempfile.TemporaryDirectory()
    exported_tracks_dir = exported_tracks_dir_instance.name

    # Segmentation and tracking
    video_path = str(Path(video_path))
    segmentation_model = str(Path(segmentation_model))
    perform_tracking(
        video_path,
        fiji_dir,
        xml_model_dir,
        segmentation_model if not default_model_check_box else None,
        fast_mode_check_box,
    )

    raw_video = re_organize_channels(img_layer.data)  # TXYC

    # Mitosis track_generation
    perform_mitosis_track_generation(
        raw_video,
        img_layer.name,
        xml_model_dir,
        exported_mitoses_dir,
        exported_tracks_dir,
    )

    # Mid-body detection
    perform_mid_body_detection(
        raw_video,
        img_layer.name,
        exported_mitoses_dir,
        exported_tracks_dir,
        movies_save_dir if save_check_box else None,
    )

    # MT cut detection
    perform_mt_cut_detection(raw_video, img_layer.name, exported_mitoses_dir)

    # Results saving
    perform_results_saving(exported_mitoses_dir, save_dir=results_save_dir)

    # Delete temporary folders
    xml_model_dir_instance.cleanup()
    exported_mitoses_dir_instance.cleanup()
    exported_tracks_dir_instance.cleanup()

    print("\nWhole process finished with success!")


@magic_factory(
    call_button="Run Segmentation and Tracking",
    layout="vertical",
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
        widget_type="CheckBox",
        text="Use default segmentation model?",
        value=True,
    ),
    segmentation_model=dict(
        widget_type="FileEdit",
        label="If not checked, cellpose segmentation model: ",
    ),
    fast_mode_check_box=dict(
        widget_type="CheckBox", text="Enable fast mode?", value=False
    ),
)
def segmentation_tracking(
    img_viewer: "napari.Viewer",
    fiji_dir: str,
    xml_model_dir: str,
    default_model_check_box: str,
    segmentation_model: str,
    fast_mode_check_box: bool,
):
    # Make sure there is only one image loaded as no layer has to be provided
    assert len(img_viewer.layers) == 1
    # Convert video path from windows to linux
    video_path = img_viewer.layers[0].source.path
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
    mitoses_save_dir: Optional[str],
    tracks_save_dir: Optional[str],
):
    raw_video = re_organize_channels(img_layer.data)  # TXYC
    perform_mitosis_track_generation(
        raw_video,
        img_layer.name,
        xml_model_dir,
        mitoses_save_dir,
        tracks_save_dir,
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
    save_check_box=dict(
        widget_type="CheckBox", text="Save cell divisions movies?", value=False
    ),
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
def micro_tubules_cut_detection(
    img_layer: "napari.layers.Image", exported_mitoses_dir: str
):
    raw_video = re_organize_channels(img_layer.data)  # TXYC
    perform_mt_cut_detection(
        raw_video,
        img_layer.name,
        exported_mitoses_dir,
    )
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
        label="Directory to save results: ",
        mode="d",
    ),
)
def results_saving(
    exported_mitoses_dir: str,
    results_save_dir: str,
):
    perform_results_saving(exported_mitoses_dir, save_dir=results_save_dir)
    print("\nResults saved with success!")
