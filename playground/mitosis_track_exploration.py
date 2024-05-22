import pickle
import os
from os.path import join
from typing import Optional
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from aicsimageio.aics_image import AICSImage

from cut_detector.utils.mitosis_track import MitosisTrack
from cut_detector.data.tools import get_data_path

PLOT_ONLY_WRONG = False


def main(
    mitoses_path: Optional[str] = get_data_path("mitoses"),
    target_name: Optional[str] = None,
):
    """Playground function to plot mid-body detections vs ground-truth detections."""
    with os.scandir(mitoses_path) as it:
        if isinstance(target_name, str):
            files = [
                e
                for e in it
                if e.is_file(follow_symlinks=False) and e.name == target_name
            ]
        else:
            files = [
                e
                for e in it
                if e.is_file(follow_symlinks=False) and e.name.endswith(".bin")
            ]

    for f in files:
        video_name = f.name.split("_mitosis_")[0]
        plot_track(f.path, video_name, f.name)


def plot_track(path: str, dirname: str, filename: str):
    """Main plotting function."""
    with open(path, "rb") as f:
        track: MitosisTrack = pickle.load(f)
        track.adapt_deprecated_attributes()

    print("loaded:", filename)

    if track.gt_mid_body_spots is None:
        print(f"no gt for {path}")
        return
    if len(track.daughter_track_ids) != 1:
        print(f"Invalid division for {path}")
        return

    (
        is_correctly_detected,
        percent_detected,
        average_position_difference,
    ) = track.evaluate_mid_body_detection(avg_as_int=False)

    if PLOT_ONLY_WRONG and is_correctly_detected:
        print(f"Mid-body correctly detected for {filename}")
        return

    bin_path = Path(path)
    tiff_path = bin_path.parent.parent / Path(
        join("mitosis_movies", f"{bin_path.stem}.tiff")
    )
    img = read_tiff(tiff_path)

    ymax = img.shape[3]
    xmax = img.shape[4]
    print("x", xmax, "y", ymax)

    gt_frame_start = min(track.gt_mid_body_spots.keys())
    gt_frame_end = max(track.gt_mid_body_spots.keys())
    gt_full_frames = range(gt_frame_start, gt_frame_end + 1)

    ordered_gt_spots = [
        track.gt_mid_body_spots.get(f) or None for f in gt_full_frames
    ]
    ordered_test_spots = [
        track.mid_body_spots.get(f) or None for f in gt_full_frames
    ]

    assert len(ordered_gt_spots) == len(ordered_test_spots)

    for s, frame in zip(ordered_gt_spots, gt_full_frames):
        print(f"f:{frame - track.min_frame} x:{s.x} y:{s.y}")

    gt_max_frame = (
        track.gt_key_events_frame["second_mt_cut"]
        if "second_mt_cut" in track.gt_key_events_frame
        else max(track.gt_mid_body_spots.keys())
    )
    eval_frame_range_sym = range(
        track.gt_key_events_frame["cytokinesis"], gt_max_frame - 1
    )

    for frame, cur_gt, next_gt, cur_f, next_f in zip(
        gt_full_frames,
        ordered_gt_spots[:-1],
        ordered_gt_spots[1:],
        ordered_test_spots[:-1],
        ordered_test_spots[1:],
    ):

        if frame in eval_frame_range_sym:
            test_fmt = "o-b"
            gt_fmt = "o-g"
            alpha = 0.4
        else:
            test_fmt = ".--b"
            gt_fmt = ".--g"
            alpha = 0.1

        if cur_gt is not None and next_gt is not None:
            plt.plot(
                [cur_gt.x, next_gt.x],
                [cur_gt.y, next_gt.y],
                gt_fmt,
                alpha=alpha,
            )
            plt.text(
                cur_gt.x,
                cur_gt.y,
                str(frame - track.min_frame),
                color="magenta",
                fontsize=6,
            )
        if cur_f is not None and next_f is not None:
            plt.plot(
                [cur_f.x, next_f.x], [cur_f.y, next_f.y], test_fmt, alpha=alpha
            )
            plt.text(
                cur_f.x,
                cur_f.y,
                str(frame - track.min_frame),
                color="red",
                fontsize=8,
            )

    plt.gca().set_xlim(0, xmax)
    plt.gca().set_ylim(ymax, 0)

    plt.title(
        f"Correct:{is_correctly_detected} | %detec:{percent_detected} | avgdiff:{average_position_difference:.3f}"
    )
    plt.suptitle(f"{dirname}:{filename}")

    plt.show()


def read_tiff(path: str) -> np.ndarray:
    """Duplicated function from cnn_framework.
    Rewritten here to avoid long useless imports.
    """
    aics_img = AICSImage(path)
    target_order = "TCZYX"
    original_order = aics_img.dims.order

    img = aics_img.data

    # Add missing dimensions if necessary
    for dim in target_order:
        if dim not in original_order:
            original_order = dim + original_order
            img = np.expand_dims(img, axis=0)

    indexes = [original_order.index(dim) for dim in target_order]

    return np.moveaxis(img, indexes, list(range(len(target_order))))


if __name__ == "__main__":
    DEFAULT_RUN = True

    if DEFAULT_RUN:
        main()
    else:
        # Custom paths for testing
        # Careful, folder must contain a folder "mitosis_movies" with tiff files
        DIRECTORY = "std"
        DIRECTORY_MAPPING = {
            "std": "Data Standard",
            "spas": "Data spastin",
            "cep": "Data cep55",
        }

        TARGET_NAME = "converted t2_t3_F-1E5-35-12_mitosis_4_13_to_189.bin"

        main(mitoses_path=f"eval_data/{DIRECTORY_MAPPING[DIRECTORY]}/mitoses/")
