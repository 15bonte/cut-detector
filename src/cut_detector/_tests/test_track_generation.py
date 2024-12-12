import os
from skimage import io

from cut_detector._widget import mitosis_track_generation
from cut_detector.data.tools import get_data_path
from cut_detector.widget_functions.mitosis_track_generation import (
    perform_mitosis_track_generation,
)


def test_open_track_generation_widget():
    """Test opening the track generation widget."""
    mitosis_track_generation()


def test_track_generation():
    """Test track generation."""

    video = io.imread(
        os.path.join(get_data_path("videos"), "example_video.tif")
    )  # TYXC

    mitosis_tracks, _, _ = perform_mitosis_track_generation(
        video,
        "example_video",
        get_data_path("spots"),
        get_data_path("tracks"),
        save=False,
    )

    assert len(mitosis_tracks) == 1

    mitosis_track = mitosis_tracks[0]

    assert mitosis_track.daughter_track_ids == [5]
    assert mitosis_track.mother_track_id == 0
    assert 7 <= mitosis_track.key_events_frame["metaphase"] <= 9  # should be 8
    assert (
        10 <= mitosis_track.key_events_frame["no_mt_cut"] <= 12
    )  # beginning of cytokinesis - should be 11
