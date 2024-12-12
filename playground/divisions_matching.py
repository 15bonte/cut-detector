"""Playground to perform divisions matching between cut detector and manual divisions."""

from cut_detector.widget_functions.divisions_matching import (
    perform_divisions_matching,
)


def main(cut_detector_file: str, folder_manual: str) -> None:
    perform_divisions_matching(
        cut_detector_file,
        folder_manual,
        maximum_frame_distance=5,
        maximum_position_distance=10,
    )


if __name__ == "__main__":
    main(
        r"C:\Users\thoma\OneDrive\Bureau\results\results\results.csv",
        r"C:\Users\thoma\OneDrive\Bureau\results\results - ground truth",
    )
