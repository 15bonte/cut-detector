import os
from skimage import io

from cut_detector._widget import mid_body_detection
from cut_detector.data.tools import get_data_path
from cut_detector.widget_functions.mid_body_detection import (
    perform_mid_body_detection,
)


def test_open_mid_body_detection_widget():
    # Just try to open the widget
    mid_body_detection()


def test_mid_body_detection_widget():

    # Add video
    video = io.imread(
        os.path.join(get_data_path("videos"), "example_video.tif")
    )  # TYXC

    # Run process
    perform_mid_body_detection(
        video,
        "example_video",
        get_data_path("mitoses"),
        get_data_path("tracks"),
        save_dir=None,
        update_mitoses=False,
    )
