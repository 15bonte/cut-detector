from pathlib import Path
from typing import Optional

from magicgui import magic_factory
import tempfile

import numpy as np
from skimage import io


from .utils.tools import re_organize_channels

from .widget_functions.tracking import perform_tracking
from .widget_functions.mid_body_detection import perform_mid_body_detection
from .widget_functions.mitosis_track_generation import (
    perform_mitosis_track_generation,
)
from .widget_functions.mt_cut_detection import perform_mt_cut_detection
from .widget_functions.save_results import perform_results_saving


def video_whole_process(
    video: np.ndarray,
    video_name: np.ndarray,
    default_model_check_box: bool,
    segmentation_model: str,
    save_check_box: bool,
    movies_save_dir: str,
    spots_dir_name: str,
    tracks_dir_name: str,
    mitoses_dir_name: str,
) -> None:
    """Perform the whole process on a single video."""

    perform_tracking(
        video,
        str(Path(segmentation_model)) if not default_model_check_box else None,
        video_name,
        spots_dir_name,
        tracks_dir_name,
    )
    perform_mitosis_track_generation(
        video,
        video_name,
        spots_dir_name,
        tracks_dir_name,
        mitoses_dir_name,
    )
    perform_mid_body_detection(
        video,
        video_name,
        mitoses_dir_name,
        tracks_dir_name,
        movies_save_dir if save_check_box else None,
    )
    perform_mt_cut_detection(video, video_name, mitoses_dir_name)


@magic_factory(
    call_button="Run Whole Process (Single Video)",
    layout="vertical",
    default_model_check_box=dict(
        widget_type="CheckBox",
        text="Use default segmentation model?",
        value=True,
    ),
    segmentation_model=dict(
        widget_type="FileEdit",
        label="If not checked, cellpose segmentation model: ",
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
    default_model_check_box: bool,
    segmentation_model: str,
    save_check_box: bool,
    movies_save_dir: str,
    results_save_dir: str,
):

    # Create temporary folders
    spots_dir = tempfile.TemporaryDirectory()
    tracks_dir = tempfile.TemporaryDirectory()
    mitoses_dir = tempfile.TemporaryDirectory()

    video = re_organize_channels(img_layer.data)  # TXYC

    video_whole_process(
        video,
        img_layer.name,
        default_model_check_box,
        segmentation_model,
        save_check_box,
        movies_save_dir,
        spots_dir.name,
        tracks_dir.name,
        mitoses_dir.name,
    )

    # Results saving
    perform_results_saving(mitoses_dir.name, save_dir=results_save_dir)

    # Delete temporary folders
    spots_dir.cleanup()
    tracks_dir.cleanup()
    mitoses_dir.cleanup()

    print("\nWhole process finished with success!")


@magic_factory(
    call_button="Run Whole Process (Folder)",
    layout="vertical",
    raw_data_dir=dict(
        widget_type="FileEdit",
        label="Data folder: ",
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
def whole_process_folder(
    raw_data_dir: str,
    default_model_check_box: bool,
    segmentation_model: str,
    save_check_box: bool,
    movies_save_dir: str,
    results_save_dir: str,
):

    # Create temporary folders
    spots_dir = tempfile.TemporaryDirectory()
    tracks_dir = tempfile.TemporaryDirectory()
    mitoses_dir = tempfile.TemporaryDirectory()

    # Run process on each video
    tiff_files = list(Path(raw_data_dir).rglob("*.tif"))
    for tiff_file in tiff_files:
        video = io.imread(tiff_file)  # TYXC
        video_whole_process(
            video,
            tiff_file.stem,
            default_model_check_box,
            segmentation_model,
            save_check_box,
            movies_save_dir,
            spots_dir.name,
            tracks_dir.name,
            mitoses_dir.name,
        )

    # Results saving
    perform_results_saving(mitoses_dir.name, save_dir=results_save_dir)

    # Delete temporary folders
    spots_dir.cleanup()
    tracks_dir.cleanup()
    mitoses_dir.cleanup()

    print("\nWhole process finished with success!")


@magic_factory(
    call_button="Run Segmentation and Tracking",
    layout="vertical",
    spots_save_dir=dict(
        widget_type="FileEdit",
        label="Directory to save .bin cell spots: ",
        mode="d",
    ),
    tracks_save_dir=dict(
        widget_type="FileEdit",
        label="Directory to save .bin cell tracks: ",
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
)
def segmentation_tracking(
    img_layer: "napari.layers.Image",
    spots_save_dir: str,
    tracks_save_dir: str,
    default_model_check_box: bool,
    segmentation_model: str,
):

    raw_video = re_organize_channels(img_layer.data)  # TXYC

    # Segmentation and tracking
    perform_tracking(
        raw_video,
        str(Path(segmentation_model)) if not default_model_check_box else None,
        img_layer.name,
        spots_save_dir,
        tracks_save_dir,
    )

    print("\nSegmentation and tracking finished with success!")


@magic_factory(
    call_button="Run Mitosis Track Generation",
    layout="vertical",
    spots_save_dir=dict(
        widget_type="FileEdit",
        label="Directory to save .bin cell spots: ",
        mode="d",
    ),
    tracks_save_dir=dict(
        widget_type="FileEdit",
        label="Directory to save .bin cell tracks: ",
        mode="d",
    ),
    mitoses_save_dir=dict(
        widget_type="FileEdit",
        label="Directory to save .bin mitoses: ",
        mode="d",
    ),
)
def mitosis_track_generation(
    img_layer: "napari.layers.Image",
    spots_save_dir: str,
    tracks_save_dir: str,
    mitoses_save_dir: Optional[str],
):
    raw_video = re_organize_channels(img_layer.data)  # TXYC

    perform_mitosis_track_generation(
        raw_video,
        img_layer.name,
        spots_save_dir,
        tracks_save_dir,
        mitoses_save_dir,
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
    movies_save_dir=dict(
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
    movies_save_dir: str,
):
    raw_video = re_organize_channels(img_layer.data)  # TXYC
    perform_mid_body_detection(
        raw_video,
        img_layer.name,
        exported_mitoses_dir,
        exported_tracks_dir,
        movies_save_dir if save_check_box else None,
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
