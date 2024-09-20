import os
from typing import Optional
import pickle
from skimage import io
import napari

from cut_detector.data.tools import get_data_path
from cut_detector.factories.results_saving_factory import ResultsSavingFactory
from cut_detector.utils.cell_track import CellTrack
from cut_detector.utils.mitosis_track import MitosisTrack


def main(
    image_path: Optional[str] = os.path.join(
        get_data_path("videos"), "example_video.tif"
    ),
    mitoses_path: Optional[str] = get_data_path("mitoses"),
    segmentation_results_path: Optional[str] = os.path.join(
        get_data_path("segmentation_results"), "example_video.bin"
    ),
    tracks_dir: Optional[str] = get_data_path("tracks"),
    video_name="example_video",
) -> None:
    """
    Parameters
    ----------
    image_path : str
        Path to the image to display, by default example_video.tif.
    mitoses_path : str
        Path to the mitoses, by default mitoses.
    segmentation_results_path : str
        Path to the segmentation results, by default example_video.bin.
    """

    # Create a Napari viewer
    viewer = napari.Viewer()

    # Load Cellpose results
    with open(segmentation_results_path, "rb") as f:
        cellpose_results = pickle.load(f)  # TYX

    # Load cell tracks
    cell_tracks: list[CellTrack] = []
    video_tracks_save_dir = os.path.join(tracks_dir, video_name)
    for state_path in os.listdir(video_tracks_save_dir):
        with open(os.path.join(video_tracks_save_dir, state_path), "rb") as f:
            cell_track = CellTrack.load(f)
            cell_tracks.append(cell_track)

    # Add video
    video = io.imread(image_path)  # TYXC

    # Match Napari video display
    viewer.add_image(video, name="example_video", rgb=True)

    # Load mitosis tracks
    mitosis_tracks = []
    for state_path in os.listdir(mitoses_path):
        with open(os.path.join(mitoses_path, state_path), "rb") as f:
            mitosis_track = MitosisTrack.load(f)
        mitosis_tracks.append(mitosis_track)

    ResultsSavingFactory().generate_napari_tracking_mask(
        mitosis_tracks,
        video,
        viewer,
        segmentation_results=cellpose_results,
        cell_tracks=cell_tracks,
    )

    napari.run()


if __name__ == "__main__":
    main()
