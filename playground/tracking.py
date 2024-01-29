import os
import pickle
from typing import Optional

from cut_detector.data.tools import get_data_path


def load_tracks_and_spots():
    trackmate_tracks = []
    for spot_file in os.listdir(trackmate_tracks_path):
        with open(os.path.join(trackmate_tracks_path, spot_file), "rb") as f:
            trackmate_tracks.append(pickle.load(f))

    spots = []
    for spot_file in os.listdir(spots_path):
        with open(os.path.join(spots_path, spot_file), "rb") as f:
            spots.append(pickle.load(f))


def main(
    segmentation_results_path: Optional[str] = os.path.join(
        get_data_path("segmentation_results"), "example_video.bin"
    ),
    trackmate_tracks_path: Optional[str] = os.path.join(
        get_data_path("tracks"), "example_video"
    ),
    spots_path: Optional[str] = os.path.join(
        get_data_path("spots"), "example_video"
    ),
):
    # Load Cellpose results
    with open(segmentation_results_path, "rb") as f:
        cellpose_results = pickle.load(f)

    # TODO: perform tracking using laptrack

    # Load TrackMate results to compare... make sure they match!
    trackmate_tracks = []
    for spot_file in os.listdir(trackmate_tracks_path):
        with open(os.path.join(trackmate_tracks_path, spot_file), "rb") as f:
            trackmate_tracks.append(pickle.load(f))

    spots = []
    for spot_file in os.listdir(spots_path):
        with open(os.path.join(spots_path, spot_file), "rb") as f:
            spots.append(pickle.load(f))

    # Load TrackMate results to compare... make sure they match!
    load_tracks_and_spots()

    # Read useful information from xml file
    tracks_merging_factory = TracksMergingFactory()
    xml_model_path = os.path.join(xml_model_dir, f"{video_name}_model.xml")
    trackmate_tracks, raw_spots = tracks_merging_factory.read_trackmate_xml(
        xml_model_path, raw_video.shape
    )


if __name__ == "__main__":
    main()
