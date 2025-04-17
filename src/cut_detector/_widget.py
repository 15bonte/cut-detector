import os
from pathlib import Path
import shutil
import time
from typing import Optional

from magicgui import magic_factory
import tempfile

import numpy as np
from skimage import io

from .utils.cell_track import CellTrack
from .utils.parameters import Parameters
from .utils.tools import re_organize_channels

from .widget_functions.segmentation_tracking import perform_tracking
from .widget_functions.mid_body_detection import perform_mid_body_detection
from .widget_functions.mitosis_track_generation import (
    perform_mitosis_track_generation,
)
from .widget_functions.mt_cut_detection import perform_mt_cut_detection
from .widget_functions.save_results import (
    perform_results_saving,
    save_galleries,
)
from .widget_functions.divisions_matching import (
    perform_divisions_matching,
)
from .widget_functions.distribution_comparison import perform_distribution_comparison


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
    results_save_dir: str,
    params=Parameters(),
) -> tuple[list[CellTrack], np.ndarray]:
    """Perform the whole process on a single video.

    Parameters
    ----------
    video : np.ndarray
        Video. TYXC.
    video_name : str
        Video name.
    default_model_check_box : bool
        Use default segmentation model?
    segmentation_model : str
        Cellpose segmentation model.
    save_check_box : bool
        Save cell divisions movies?
    movies_save_dir : str
        Directory to save division movies.
    spots_dir_name : str
        Directory to save .bin cell spots.
    tracks_dir_name : str
        Directory to save .bin cell tracks.
    mitoses_dir_name : str
        Directory to save .bin mitoses.
    results_save_dir : str
        Directory to save results.
    params : Parameters, optional
        Video parameters.

    Returns
    -------
    list[CellTrack]
        Cell tracks and segmentation results.
    np.ndarray
        Segmentation results.
    """

    _, cell_tracks, segmentation_results = perform_tracking(
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
        params=params,
    )
    perform_mid_body_detection(
        video,
        video_name,
        mitoses_dir_name,
        tracks_dir_name,
        movies_save_dir if save_check_box else None,
        parallel_detection=True,
        params=params,
    )
    perform_mt_cut_detection(
        video, video_name, mitoses_dir_name, params=params
    )
    save_galleries(video, video_name, mitoses_dir_name, results_save_dir)

    return cell_tracks, segmentation_results


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
    debug_mode_check_box=dict(
        widget_type="CheckBox",
        text="Debug mode",
        value=False,
    ),
    display_check_box=dict(
        widget_type="CheckBox",
        text="Display segmentation and tracking",
        value=False,
    ),
)
def whole_process(
    img_layer: "napari.layers.Image",
    viewer: "napari.Viewer",
    default_model_check_box: bool,
    segmentation_model: str,
    save_check_box: bool,
    movies_save_dir: str,
    results_save_dir: str,
    debug_mode_check_box: bool,
    display_check_box: bool,
):

    start = time.time()

    params = Parameters()

    # Create temporary folders
    spots_dir = tempfile.TemporaryDirectory()
    tracks_dir = tempfile.TemporaryDirectory()
    mitoses_dir = tempfile.TemporaryDirectory()

    video = re_organize_channels(img_layer.data)  # TYXC

    cell_tracks, segmentation_results = video_whole_process(
        video,
        img_layer.name,
        default_model_check_box,
        segmentation_model,
        save_check_box,
        movies_save_dir,
        spots_dir.name,
        tracks_dir.name,
        mitoses_dir.name,
        results_save_dir,
        params=params,
    )

    end = time.time()
    print(f"\nProcess finished in {int((end - start) / 60)} minutes!")

    # Results saving
    if display_check_box:
        perform_results_saving(
            mitoses_dir.name,
            save_dir=results_save_dir,
            video=img_layer.data,
            viewer=viewer,
            segmentation_results=segmentation_results,
            cell_tracks=cell_tracks,
        )
    else:
        perform_results_saving(
            mitoses_dir.name,
            save_dir=results_save_dir,
            video=img_layer.data,
            viewer=viewer,
        )

    if debug_mode_check_box:
        shutil.copytree(
            spots_dir.name, os.path.join(results_save_dir, "spots")
        )
        shutil.copytree(
            tracks_dir.name, os.path.join(results_save_dir, "tracks")
        )
        shutil.copytree(
            mitoses_dir.name, os.path.join(results_save_dir, "mitoses")
        )

    # Delete temporary folders
    spots_dir.cleanup()
    tracks_dir.cleanup()
    mitoses_dir.cleanup()


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

    params = Parameters()

    # Run process on each video
    tiff_files = list(Path(raw_data_dir).rglob("*.tif"))
    for i, tiff_file in enumerate(tiff_files):
        print(
            f"\nProcessing {tiff_file.stem} - Video {i+1}/{len(tiff_files)}\n"
        )
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
            results_save_dir,
            params,
        )

    # Results saving
    perform_results_saving(mitoses_dir.name, save_dir=results_save_dir)

    # Delete temporary folders
    spots_dir.cleanup()
    tracks_dir.cleanup()
    mitoses_dir.cleanup()

    print("\nProcess finished with success!")


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

    raw_video = re_organize_channels(img_layer.data)  # TYXC

    # Segmentation and tracking
    perform_tracking(
        raw_video,
        str(Path(segmentation_model)) if not default_model_check_box else None,
        img_layer.name,
        spots_save_dir,
        tracks_save_dir,
    )


@magic_factory(
    call_button="Run Mitosis Track Generation",
    layout="vertical",
    spots_load_dir=dict(
        widget_type="FileEdit",
        label="Directory to load .bin cell spots: ",
        mode="d",
    ),
    tracks_load_dir=dict(
        widget_type="FileEdit",
        label="Directory to load .bin cell tracks: ",
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
    spots_load_dir: str,
    tracks_load_dir: str,
    mitoses_save_dir: Optional[str],
):
    raw_video = re_organize_channels(img_layer.data)  # TYXC

    params = Parameters()

    perform_mitosis_track_generation(
        raw_video,
        img_layer.name,
        spots_load_dir,
        tracks_load_dir,
        mitoses_save_dir,
        params=params,
    )


@magic_factory(
    call_button="Run Mid-body Detection",
    layout="vertical",
    exported_mitoses_dir=dict(
        widget_type="FileEdit",
        label="Directory to load .bin mitoses:",
        mode="d",
    ),
    exported_tracks_dir=dict(
        widget_type="FileEdit",
        label="Directory to load .bin cell tracks: ",
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
    params = Parameters()

    raw_video = re_organize_channels(img_layer.data)  # TYXC
    perform_mid_body_detection(
        raw_video,
        img_layer.name,
        exported_mitoses_dir,
        exported_tracks_dir,
        movies_save_dir if save_check_box else None,
        parallel_detection=True,
        params=params,
    )


@magic_factory(
    call_button="Run MT Cut Detection",
    layout="vertical",
    exported_mitoses_dir=dict(
        widget_type="FileEdit",
        label="Directory to load .bin mitoses: ",
        mode="d",
    ),
)
def micro_tubules_cut_detection(
    img_layer: "napari.layers.Image", exported_mitoses_dir: str
):
    raw_video = re_organize_channels(img_layer.data)  # TYXC
    params = Parameters()
    perform_mt_cut_detection(
        raw_video,
        img_layer.name,
        exported_mitoses_dir,
        params=params,
    )


@magic_factory(
    call_button="Save results",
    layout="vertical",
    exported_mitoses_dir=dict(
        widget_type="FileEdit",
        label="Directory to load .bin mitoses: ",
        mode="d",
    ),
    exported_tracks_dir=dict(
        widget_type="FileEdit",
        label="Directory to load .bin tracks: ",
        mode="d",
    ),
    results_save_dir=dict(
        widget_type="FileEdit",
        label="Directory to save results: ",
        mode="d",
    ),
)
def results_saving(
    img_layer: "napari.layers.Image",
    viewer: "napari.Viewer",
    exported_mitoses_dir: str,
    exported_tracks_dir: str,
    results_save_dir: str,
):
    params = Parameters()
    # Load cell tracks
    cell_tracks: list[CellTrack] = []
    # Iterate over "bin" files in exported_tracks_dir
    video_exported_tracks_dir = os.path.join(
        exported_tracks_dir, img_layer.name
    )
    for state_path in os.listdir(video_exported_tracks_dir):
        # Load mitosis track
        with open(
            os.path.join(video_exported_tracks_dir, state_path), "rb"
        ) as f:
            cell_track = CellTrack.load(f)
            cell_tracks.append(cell_track)

    perform_results_saving(
        exported_mitoses_dir,
        save_dir=results_save_dir,
        video=img_layer.data,
        viewer=viewer,
        cell_tracks=cell_tracks,
        params=params,
    )


@magic_factory(
    call_button="Match divisions",
    layout="vertical",
    requirements=dict(
        widget_type="Label",
        label="Requirements",
        value="Manual annotation files must contain the following columns:\n - Video name ('video') \n - Cytokinesis onset frame ('cyto')\n - MT cut frame ('cut')\n - Midbody position x ('x')\n - Midbody position y ('y')\n \n Video names must match exactly.\n",
    ),
    cut_detector_file=dict(
        widget_type="FileEdit",
        label="Cut Detector results (csv): ",
    ),
    folder_manual=dict(
        widget_type="FileEdit",
        label="Manual annotations folder: ",
        mode="d",
    ),
    maximum_frame_distance=dict(
        widget_type="SpinBox",
        label="Max cytokinesis frame difference: ",
        value=3,  # Default value
    ),
    maximum_position_distance=dict(
        widget_type="SpinBox",
        label="Max midbody position distance: ",
        value=50,  # Default value
    ),
)
def divisions_matching(
    requirements: str,
    cut_detector_file: str,
    folder_manual: str,
    maximum_frame_distance: str,
    maximum_position_distance: str,
):
    perform_divisions_matching(
        str(Path(cut_detector_file)),
        folder_manual,
        maximum_frame_distance,
        maximum_position_distance,
    )

@magic_factory(
    call_button="Compare distributions",
    layout="vertical",
    requirements=dict(
        widget_type="Label",
        label="Requirements",
        value="Matched Cut Detector results is the output from the 'Match divisions'.\nManual annotations file is the same as in this previous widget.\nImportant note: add a first column to both files with the name of the experiment.",
    ),
    matched_csv=dict(
        widget_type="FileEdit",
        label="Matched Cut Detector results (csv): ",
    ),
    manual_csv=dict(
        widget_type="FileEdit",
        label="Manual annotations file (csv): ",
    ),
    save_folder=dict(
        widget_type="FileEdit",
        label="Directory to save results: ",
        mode="d",
    ),
    max_cytokinesis_frame=dict(
        widget_type="SpinBox",
        label="Max frame for cytokinesis onset: ",
        value=216,  # Default value
    ),
    conditions_vs_manual=dict(
        widget_type="ListEdit",
        label="List of conditions to compare with manual annotations: ",
        value=["Control"],  # Default value
    ),   
    conditions_vs_control=dict(
        widget_type="ListEdit",
        label="List of conditions to compare to control: ",
        value=[],  # Default value
    ),   
    deltas=dict(
        widget_type="ListEdit",
        label="List of deltas to consider for equivalence tests: ",
        value=[10, 20, 30, 40],  # Default value
    ),   
)
def distribution_comparison(
    requirements: str,
    matched_csv: str,
    manual_csv: str,
    save_folder: str,
    max_cytokinesis_frame: int,
    conditions_vs_manual: list[str],
    conditions_vs_control: list[str],
    deltas: list[int],
):
    perform_distribution_comparison(
        matched_csv,
        manual_csv,
        save_folder,
        max_cytokinesis_frame,
        conditions_vs_manual,
        conditions_vs_control,
        deltas
    )
