import os
from skimage import io

from cut_detector._widget import micro_tubules_cut_detection
from cut_detector.data.tools import get_data_path
from cut_detector.widget_functions.mt_cut_detection import (
    perform_mt_cut_detection,
)


def test_open_micro_tubules_cut_detection_widget():
    # Just try to open the widget
    micro_tubules_cut_detection()


def test_micro_tubules_cut_detection():

    # Add video
    video = io.imread(
        os.path.join(get_data_path("videos"), "example_video.tif")  # TYXC
    )

    perform_mt_cut_detection(
        video,
        "example_video",
        get_data_path("mitoses"),
        update_mitoses=False,
    )
