import os
import pickle
from typing import Optional
import numpy as np
from ..utils.cell_spot import CellSpot
from ..utils.cell_track import CellTrack
from ..factories.segmentation_tracking_factory import (
    SegmentationTrackingFactory,
)
from ..models.tools import get_model_path


def perform_tracking(
    video: np.ndarray,
    model_path: Optional[str] = None,
    video_name: Optional[str] = None,
    spots_save_dir: Optional[str] = None,
    tracks_save_dir: Optional[str] = None,
    save: bool = True,
) -> tuple[list[CellSpot], list[CellTrack], np.ndarray]:
    """
    Run cell segmentation and tracking on the given video.

    Parameters
    ----------
    video : np.ndarray, TYXC
        The video to run the segmentation and tracking on.
    model_path : str, optional
        Path to the cellpose model, by default None.
    video_name : str, optional
        Name of the video, by default None.
    spots_save_dir : str, optional
        Directory to save cell spots, by default None.
    tracks_save_dir : str, optional
        Directory to save cell tracks, by default None.
    save : bool, optional
        Save the results, by default True.

    Returns
    -------
    tuple[list[CellSpot], list[CellTrack], np.ndarray]
        List of cell spots, list of cell tracks, segmentation results.
    """

    print("### SEGMENTATION & TRACKING ###")

    if model_path is None:
        model_path = os.path.join(
            get_model_path("segmentation"), "segmentation_model"
        )

    segmentation_tracking_factory = SegmentationTrackingFactory(model_path)
    # Swap dimensions from TYXC to TCYX
    video = np.moveaxis(video, 3, 1)
    cell_spots, cell_tracks, segmentation_results = (
        segmentation_tracking_factory.perform_segmentation_tracking(video)
    )

    if not save:
        return cell_spots, cell_tracks, segmentation_results

    assert spots_save_dir is not None
    assert tracks_save_dir is not None

    # Create saving directories if they do not exist
    video_spots_save_dir = os.path.join(spots_save_dir, video_name)
    if not os.path.exists(video_spots_save_dir):
        os.makedirs(video_spots_save_dir)
    video_tracks_save_dir = os.path.join(tracks_save_dir, video_name)
    if not os.path.exists(video_tracks_save_dir):
        os.makedirs(video_tracks_save_dir)

    # Save cell spots
    for cell_spot in cell_spots:
        state_path = f"spot_{cell_spot.id}.bin"
        save_path = os.path.join(
            video_spots_save_dir,
            state_path,
        )
        with open(save_path, "wb") as f:
            pickle.dump(cell_spot, f)

    # Save cell tracks
    for cell_track in cell_tracks:
        state_path = f"track_{cell_track.track_id}.bin"
        save_path = os.path.join(
            video_tracks_save_dir,
            state_path,
        )
        with open(save_path, "wb") as f:
            pickle.dump(cell_track, f)

    return cell_spots, cell_tracks, segmentation_results
