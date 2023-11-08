import os
from skimage import io

from cut_detector._widget import mitosis_track_generation
from cut_detector.data.tools import get_data_path


def test_open_track_generation_widget():
    # Just try to open the widget
    mitosis_track_generation()


def test_track_generation_widget(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()

    # Add video
    video = io.imread(os.path.join(get_data_path("videos"), "example_video.tif"))
    viewer.add_image(video, name="example_video")

    # Open the widget
    widget = mitosis_track_generation()

    # Run process
    widget(
        viewer.layers[0],
        get_data_path("models"),
        get_data_path("mitoses"),
        get_data_path("tracks"),
    )
