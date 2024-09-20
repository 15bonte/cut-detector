"""Basic playground to load and inspect MitosisTrack objects."""

import os
from typing import Optional

from cut_detector.data.tools import get_data_path
from cut_detector.utils.cell_track import CellTrack


def main(
    track_folder: Optional[str] = os.path.join(
        get_data_path("tracks"), "example_video"
    ),
) -> None:
    """
    Parameters
    ----------
    track_folder : str
        Path to the track bin files.
    """

    track_files = os.listdir(track_folder)[:5]

    cell_tracks: list[CellTrack] = []
    for track_file in track_files:
        with open(os.path.join(track_folder, track_file), "rb") as f:
            cell_track = CellTrack.load(f)
            cell_tracks.append(cell_track)

    print(cell_tracks[0])


if __name__ == "__main__":
    main()
