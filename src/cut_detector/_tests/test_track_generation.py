import os
from skimage import io

from cut_detector._widget import mitosis_track_generation
from cut_detector.data.tools import get_data_path
from cut_detector.utils.tools import read_trackmate_xml
from cut_detector.widget_functions.mitosis_track_generation import (
    perform_mitosis_track_generation,
)


def test_open_track_generation_widget():
    # Just try to open the widget
    mitosis_track_generation()


def test_track_generation():

    # Add video
    video = io.imread(
        os.path.join(get_data_path("videos"), "example_video.tif")
    )  # TYXC

    # Read useful information from xml file
    xml_model_path = os.path.join(
        get_data_path("models"), "example_video_model.xml"
    )
    cell_tracks, cell_spots = read_trackmate_xml(xml_model_path, video.shape)

    # Open the widget
    mitosis_tracks = perform_mitosis_track_generation(
        video, "example_video", cell_spots, cell_tracks
    )

    assert len(mitosis_tracks) == 1

    mitosis_track = mitosis_tracks[0]

    assert mitosis_track.daughter_track_ids == [0]
    assert mitosis_track.mother_track_id == 4
    assert 7 <= mitosis_track.key_events_frame["metaphase"] <= 9  # should be 8
    assert (
        10 <= mitosis_track.key_events_frame["cytokinesis"] <= 12
    )  # should be 11
