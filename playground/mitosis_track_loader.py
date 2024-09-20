"""Basic playground to load and inspect MitosisTrack objects."""

import os
from typing import Optional

from cut_detector.data.tools import get_data_path
from cut_detector.utils.mitosis_track import MitosisTrack


def main(
    mitosis_folder: Optional[str] = get_data_path("mitoses"),
) -> None:
    """
    Parameters
    ----------
    mitosis_folder : str
        Path to the mitoses bin files.
    """

    mitosis_files = os.listdir(mitosis_folder)[:5]

    mitosis_tracks = []
    for mitosis_file in mitosis_files:
        with open(os.path.join(mitosis_folder, mitosis_file), "rb") as f:
            mitosis_track = MitosisTrack.load(f)
            mitosis_tracks.append(mitosis_track)

    print(mitosis_tracks[0])


if __name__ == "__main__":
    main()
