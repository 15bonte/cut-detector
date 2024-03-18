import os
from typing import Optional
import pickle

from cut_detector.utils.mitosis_track import MitosisTrack
from cut_detector.data.tools import get_data_path


def main(mitoses_folder: Optional[str] = get_data_path("mitoses")):
    """
    Evaluate mid-body detection from annotated files.
    """
    # Load mitosis tracks
    mitoses_tracks: list[MitosisTrack] = []
    mitoses_files = os.listdir(mitoses_folder)
    for mitosis_file in mitoses_files:
        with open(os.path.join(mitoses_folder, mitosis_file), "rb") as f:
            mitosis_track: MitosisTrack = pickle.load(f)
            mitosis_track.adapt_deprecated_attributes()
            mitoses_tracks.append(mitosis_track)

    # Iterate over mitoses
    mb_detected, mb_not_detected = 0, 0
    wrong_detections = []
    for mitosis_track, mitosis_file in zip(mitoses_tracks, mitoses_files):
        # Ignore if triple division or not annotated mitosis
        if (
            len(mitosis_track.daughter_track_ids) != 1
            or mitosis_track.gt_mid_body_spots is None
        ):
            continue
        # Perform evaluation for current mitosis
        (
            is_correctly_detected,
            percent_detected,
            average_position_difference,
        ) = mitosis_track.evaluate_mid_body_detection()
        if is_correctly_detected:
            mb_detected += 1
        else:
            mb_not_detected += 1
            wrong_detections.append(
                {
                    "path": mitosis_file,
                    "percent_detected": percent_detected,
                    "average_position_difference": average_position_difference,
                }
            )

    # Print wrong detections
    if len(wrong_detections) > 0:
        print("\nWrong detections:")
        for wrong_detection in wrong_detections:
            print(
                f"{wrong_detection['path']}: detected {wrong_detection['percent_detected']}% with avg distance {wrong_detection['average_position_difference']}"
            )

    if (mb_detected + mb_not_detected) == 0:
        print("No mid-body detection evaluation possible.")
    else:
        print(
            f"\nMid-body detection evaluation: {mb_detected / (mb_detected + mb_not_detected) * 100:.2f}% | {mb_detected}/{mb_detected + mb_not_detected}"
        )


if __name__ == "__main__":
    main()
