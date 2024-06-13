import os
from skimage import io

from cut_detector._widget import mid_body_detection
from cut_detector.data.tools import get_data_path
from cut_detector.widget_functions.mid_body_detection import (
    perform_mid_body_detection,
)


def test_open_mid_body_detection_widget():
    """Test opening the mid body detection widget."""
    mid_body_detection()


def test_mid_body_detection():
    """Test mid body detection."""
    video = io.imread(
        os.path.join(get_data_path("videos"), "example_video.tif")
    )  # TYXC

    # Run normal process
    mitosis_tracks = perform_mid_body_detection(
        video,
        "example_video",
        get_data_path("mitoses"),
        get_data_path("tracks"),
        save=False,
        parallel_detection=True,
    )

    is_correctly_detected, _, _ = mitosis_tracks[
        0
    ].evaluate_mid_body_detection()
    assert is_correctly_detected

    # Perform other strategies
    perform_mid_body_detection(
        video,
        "example_video",
        get_data_path("mitoses"),
        get_data_path("tracks"),
        save=False,
        parallel_detection=False,
    )
    perform_mid_body_detection(
        video,
        "example_video",
        get_data_path("mitoses"),
        get_data_path("tracks"),
        save=False,
        parallel_detection=False,
        detection_method="h_maxima",
    )
