import pickle
import os
from os.path import join
from typing import Union, Literal
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from aicsimageio.aics_image import AICSImage

from cut_detector.utils.mitosis_track import MitosisTrack
from cut_detector.utils.mid_body_spot import MidBodySpot

DIRECTORY = "std"
DIRECTORY_MAPPING = {
    "std":  "Data Standard",
    "spas": "Data spastin",
    "cep":  "Data cep55",
}

TARGET_NAME = "converted t2_t3_F-1E5-35-12_mitosis_4_13_to_189.bin"
# TARGET_NAME = None

def main():
    with os.scandir(make_dir_path(DIRECTORY)) as it:
        if isinstance(TARGET_NAME, str):
            files = [e for e in it if e.is_file(follow_symlinks=False) and e.name == TARGET_NAME]
        else:
            files = [e for e in it if e.is_file(follow_symlinks=False) and e.name.endswith(".bin")]
    
    for f in files:
        plot_track(f.path, DIRECTORY, f.name)


def make_path(directory: str, name: str) -> str:
    return f"eval_data/{DIRECTORY_MAPPING[directory]}/mitoses/{name}"

def make_dir_path(directory: str) -> str:
    return f"eval_data/{DIRECTORY_MAPPING[directory]}/mitoses/"

def plot_track(path: str, dirname: str, filename: str):
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

    if is_correctly_detected:
        return
    
    bin_path = Path(path)
    # print("bin path parent:", bin_path.parent)
    tiff_path = bin_path.parent.parent / Path(join("movies", f"{bin_path.stem}.tiff"))
    # print("tiff path:", tiff_path)
    img = read_tiff(tiff_path)

    ymax = img.shape[3]
    xmax = img.shape[4]
    print("x", xmax, "y", ymax)

    gt_frame_start = min(track.gt_mid_body_spots.keys())
    gt_frame_end = max(track.gt_mid_body_spots.keys())
    gt_full_frames = range(gt_frame_start, gt_frame_end+1)

    ordered_gt_spots   = [track.gt_mid_body_spots.get(f) or None for f in gt_full_frames]
    ordered_test_spots = [track.mid_body_spots.get(f) or None for f in gt_full_frames]

    assert len(ordered_gt_spots) == len(ordered_test_spots)

    for s, frame in zip(ordered_gt_spots, gt_full_frames):
        print(f"f:{frame - track.min_frame} x:{s.x} y:{s.y}")

    gt_max_frame = (
        track.gt_key_events_frame["second_mt_cut"]
        if "second_mt_cut" in track.gt_key_events_frame
        else max(track.gt_mid_body_spots.keys())
    )
    eval_frame_range     = range(track.gt_key_events_frame["cytokinesis"], gt_max_frame)
    eval_frame_range_sym = range(track.gt_key_events_frame["cytokinesis"], gt_max_frame-1)

    for frame, r_frame, cur_gt, next_gt, cur_f, next_f in zip(
            gt_full_frames,
            range(len(ordered_gt_spots[:-1])),
            ordered_gt_spots[:-1], 
            ordered_gt_spots[1:], 
            ordered_test_spots[:-1], 
            ordered_test_spots[1:]):
        

        if frame in eval_frame_range_sym:
            test_fmt = "o-b"
            gt_fmt   = "o-g"
            alpha    = 0.4
        else:
            test_fmt = ".--b"
            gt_fmt   = ".--g"
            alpha    = 0.1

        if cur_gt is not None and next_gt is not None:
            plt.plot([cur_gt.x, next_gt.x], [cur_gt.y, next_gt.y], gt_fmt, alpha=alpha)
            plt.text(cur_gt.x, cur_gt.y, str(frame - track.min_frame), color="magenta", fontsize=6)
        if cur_f is not None and next_f is not None:
            plt.plot([cur_f.x, next_f.x], [cur_f.y, next_f.y], test_fmt, alpha=alpha)
            plt.text(cur_f.x, cur_f.y, str(frame - track.min_frame), color="red", fontsize=8)

    plt.gca().set_xlim(0, xmax)
    plt.gca().set_ylim(ymax, 0)

    plt.title(f"Correct:{is_correctly_detected} | %detec:{percent_detected} | avgdiff:{average_position_difference:.3f}")
    plt.suptitle(f"{dirname}:{filename}")

    plt.show()

def print_spots_and_gt(track: MitosisTrack):
    print("spots:", sep="\n")
    print_spot_dict(track.mid_body_spots)
    print("")
    print('GT spots:', sep="\n")
    print_spot_dict(track.gt_mid_body_spots)


def print_spot_dict(d: Union[dict[int, MidBodySpot], None]):
    l_d = d or {}
    for f, spot in l_d.items():
        print(f"f:{f} x:{spot.x} y:{spot.y}")

def print_spot_list(spots: MidBodySpot):
    for s in spots:
        print(f"x:{s.x} y:{s.y}")


def read_tiff(path: str) -> np.ndarray:
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

def plot_v1(track: MitosisTrack):
    gt_spots    = list(track.gt_mid_body_spots.values())
    found_spots_in_gt = [
        s 
        for f, s in track.mid_body_spots.items() if f in track.gt_mid_body_spots.keys()
    ]

    assert len(gt_spots) == len(found_spots_in_gt)

    for cur_gt, next_gt, cur_f, next_f in zip(gt_spots[:-1], gt_spots[1:], found_spots_in_gt[:-1], found_spots_in_gt[1:]):
        plt.plot([cur_gt.x, next_gt.x], [cur_gt.y, next_gt.y], "-b")
        plt.plot([cur_f.x, next_f.x], [cur_f.y, next_f.y], "-g")

    plt.show()

if __name__ == "__main__":
    main()