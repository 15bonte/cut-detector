import os
from skimage import io

from cut_detector._widget import mid_body_detection
from cut_detector.data.tools import get_data_path


def test_open_mid_body_detection_widget():
    # Just try to open the widget
    mid_body_detection()


def test_mid_body_detection_widget(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()

    # Add video
    video = io.imread(os.path.join(get_data_path("videos"), "example_video.tif"))
    viewer.add_image(video, name="example_video")

    # Open the widget
    widget = mid_body_detection()

    # Run process
    widget(viewer.layers[0], get_data_path("mitoses"), get_data_path("tracks"), False, "")
