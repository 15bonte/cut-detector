import os
from skimage import io

from cut_detector._widget import segmentation_tracking
from cut_detector.data.tools import get_data_path
from cut_detector.widget_functions.segmentation_tracking import (
    perform_tracking,
)


def test_open_track_generation_widget():
    """Test opening the track generation widget."""
    segmentation_tracking()


def test_track_generation():
    """Test track generation."""
    video = io.imread(
        os.path.join(get_data_path("videos"), "example_video.tif")
    )  # TYXC

    cell_spots, cell_tracks, _ = perform_tracking(video, save=False)

    assert 200 <= len(cell_spots) <= 300
    assert len(cell_tracks) in [5, 6]  # should be 5
