import os
from skimage import io

from cut_detector._widget import micro_tubules_cut_detection
from cut_detector.data.tools import get_data_path
from cut_detector.widget_functions.mt_cut_detection import (
    perform_mt_cut_detection,
)


def test_open_micro_tubules_cut_detection_widget():
    """Test opening the micro tubules cut detection widget."""
    micro_tubules_cut_detection()


def test_micro_tubules_cut_detection():
    """Test micro tubules cut detection."""
    video = io.imread(
        os.path.join(get_data_path("videos"), "example_video.tif")  # TYXC
    )

    mitosis_tracks = perform_mt_cut_detection(
        video,
        "example_video",
        get_data_path("mitoses"),
        save=False,
    )

    assert (
        27 <= mitosis_tracks[0].key_events_frame["first_mt_cut"] <= 31
    )  # should be 29
    assert (
        33 <= mitosis_tracks[0].key_events_frame["second_mt_cut"] <= 37
    )  # should be 35
