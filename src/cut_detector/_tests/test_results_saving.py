import os
from skimage import io

from cut_detector._widget import results_saving
from cut_detector.data.tools import get_data_path
from cut_detector.widget_functions.save_results import perform_results_saving


def test_open_results_saving_widget():
    """Test opening the results saving widget."""
    results_saving()


def test_results_saving():
    """Test results saving."""
    video = io.imread(
        os.path.join(get_data_path("videos"), "example_video.tif")
    )  # TYXC

    perform_results_saving(
        get_data_path("mitoses"),
        show=False,
        save_dir=get_data_path("results"),
        verbose=True,
        video=video,
    )
