import os
import csv

from ..utils.spot import Spot


def parse_csv(file_path: str) -> list:
    """Parse CSV file as dictionary."""
    with open(file_path, mode="r", newline="", encoding="utf-8") as file:
        csv_reader = csv.DictReader(file, delimiter=";")
        data = [row for row in csv_reader]
    return data


def save_csv(data: dict, output_file: str) -> None:
    """Save data to CSV file."""
    if not data:
        print("No data to save.")
        return

    fieldnames = data[0].keys()
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        csv_writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=";")
        csv_writer.writeheader()
        csv_writer.writerows(data)


def infer_manual_keys(manual_division: dict) -> dict[str, str]:
    """Infer manual keys from manual division.

    Parameters
    ----------
    manual_division : dict
        Manual division.

    Returns
    -------
    dict[str, str]
        Matched manual keys.
    """
    keys = manual_division.keys()
    manual_keys = {}

    for key in keys:
        if "cyto" in key:
            manual_keys["cytokinesis frame"] = key
        elif "cut" in key:
            manual_keys["cut frame"] = key
        elif "video" in key:
            manual_keys["video"] = key
        elif "x" in key:
            manual_keys["position midbody x"] = key
        elif "y" in key:
            manual_keys["position midbody y"] = key

    assert len(manual_keys) == 5

    return manual_keys


def add_matched_manual(
    cut_detector_division: dict,
    manual_divisions: list[dict],
    manual_keys: dict[str, str],
    maximum_frame_distance: int,
    maximum_position_distance: int,
) -> None:
    """Add matched manual division to cut detector division.

    Parameters
    ----------
    cut_detector_division : dict
        Cut detector division.
    manual_divisions : list[dict]
        Manual divisions.
    manual_keys : dict[str, str]
        Matched manual keys.
    maximum_frame_distance : int
        Maximum frame distance.
    maximum_position_distance : int
        Maximum position distance.
    """
    cut_detector_spot = Spot(
        frame=int(cut_detector_division["cytokinesis frame"]),
        x=int(cut_detector_division["position midbody x"]),
        y=int(cut_detector_division["position midbody y"]),
    )

    (
        matched_cytokinesis_frame,
        matched_cut_frame,
        matched_position_x,
        matched_position_y,
    ) = (-1, -1, -1, -1)

    for idx, manual_division in enumerate(manual_divisions):
        manual_division_spot = Spot(
            frame=int(manual_division[manual_keys["cytokinesis frame"]]),
            x=int(manual_division[manual_keys["position midbody x"]]),
            y=int(manual_division[manual_keys["position midbody y"]]),
        )
        # Check matching video
        if (
            manual_division[manual_keys["video"]]
            != cut_detector_division["video"]
        ):
            continue
        # Check matching frame
        if (
            cut_detector_spot.temporal_distance_to(manual_division_spot)
            > maximum_frame_distance
        ):
            continue
        # Check matching position
        if (
            cut_detector_spot.distance_to(manual_division_spot)
            > maximum_position_distance
        ):
            continue
        # Match found!
        matched_cytokinesis_frame = manual_division[
            manual_keys["cytokinesis frame"]
        ]
        matched_cut_frame = manual_division[manual_keys["cut frame"]]
        matched_position_x = manual_division[manual_keys["position midbody x"]]
        matched_position_y = manual_division[manual_keys["position midbody y"]]
        # Remove matched manual division
        manual_divisions.pop(idx)
        break

    # No match found
    cut_detector_division["matched cytokinesis frame"] = (
        matched_cytokinesis_frame
    )
    cut_detector_division["matched cut frame"] = matched_cut_frame
    cut_detector_division["matched position midbody x"] = matched_position_x
    cut_detector_division["matched position midbody y"] = matched_position_y


def perform_divisions_matching(
    cut_detector_file: str,
    folder_manual: str,
    maximum_frame_distance: int,
    maximum_position_distance: int,
) -> None:
    """Perform divisions matching between cut detector and manual divisions.

    Parameters
    ----------
    folder_cut_detector : str
        Folder with cut detector divisions.
    folder_manual : str
        Folder with manual divisions.
    """
    print("\n### DIVISIONS MATCHING ###")

    # Load annotated divisions
    manual_divisions = []
    manual_files = os.listdir(folder_manual)
    for manual_file in manual_files:
        manual_divisions.extend(
            parse_csv(os.path.join(folder_manual, manual_file))
        )
    manual_keys = infer_manual_keys(manual_divisions[0])

    data = []
    with open(
        cut_detector_file, mode="r", newline="", encoding="utf-8"
    ) as file:
        csv_reader = csv.DictReader(file, delimiter=";")
        for row in csv_reader:
            row[""] = ""  # empty column
            add_matched_manual(
                row,
                manual_divisions,
                manual_keys,
                maximum_frame_distance,
                maximum_position_distance,
            )
            data.append(row)

    save_csv(data, cut_detector_file.replace(".csv", "_matched.csv"))
    print("Matched divisions file saved with success.")
