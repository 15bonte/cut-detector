""" Checking the state of the mitosis tracks
"""

import os
import pickle
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional

from cut_detector.data.tools import get_data_path
from cut_detector.utils.mitosis_track import MitosisTrack


@dataclass
class TrackDirStateStat:
    correct: int
    no_gt: int
    invalid_div: int
    wrong: int

    def correct_pct(self) -> float:
        return self.correct / (
            self.correct + self.no_gt + self.invalid_div + self.wrong
        )

    def no_gt_pct(self) -> float:
        return self.no_gt / (
            self.correct + self.no_gt + self.invalid_div + self.wrong
        )

    def invalid_div_pct(self) -> float:
        return self.invalid_div / (
            self.correct + self.no_gt + self.invalid_div + self.wrong
        )

    def wrong_pct(self) -> float:
        return self.wrong / (
            self.correct + self.no_gt + self.invalid_div + self.wrong
        )

    def available_correct_pct(self) -> float:
        return self.correct / (self.correct + self.wrong)


class TrackState(Enum):
    Correct = auto()
    NoGT = auto()
    InvalidDiv = auto()
    WrongResult = auto()

    def __str__(self) -> str:
        if self == TrackState.Correct:
            return "Correct"
        elif self == TrackState.NoGT:
            return "No ground truth"
        elif self == TrackState.InvalidDiv:
            return "Invalid div"
        elif self == TrackState.WrongResult:
            return "Wrong result"
        else:
            raise RuntimeError(f"No str for {self}")


def main(
    mitoses_path: Optional[str] = get_data_path("mitoses"),
    name_filter: Optional[str] = None,
):
    """Playground function to have a look at current detection accuracy.
    No detection is performed here, only the state of the tracks is checked."""
    with os.scandir(mitoses_path) as it:
        entries = [
            e
            for e in it
            if e.is_file(follow_symlinks=False) and e.name.endswith(".bin")
        ]

    states = {}

    for e in entries:
        with open(e.path, "rb") as f:
            track: MitosisTrack = pickle.load(f)
            states[e.name] = check_track(track)

    print("dir:", mitoses_path)
    print("-----")
    if isinstance(name_filter, str):
        for n, s in states.items():
            if name_filter in n:
                print(f"{n}: {s:>30}")
    else:
        for n, s in states.items():
            print(f"{n}: {s:>20}")

    stats = make_stats(states, name_filter)
    print("")
    print("")
    print(f"summary {mitoses_path} with name filter: {name_filter}")
    print("------")
    print("correct:", stats.correct, f"({stats.correct_pct()*100:.2f}%)")
    print("no_gt:", stats.no_gt, f"({stats.no_gt_pct()*100:.2f}%)")
    print(
        "invalid_div:",
        stats.invalid_div,
        f"({stats.invalid_div_pct()*100:.2f}%)",
    )
    print("wrong:", stats.wrong, f"({stats.wrong_pct()*100:.2f}%)")
    print("------")
    print("available correct:", f"({stats.available_correct_pct()*100:.2f}%)")


def check_track(track: MitosisTrack) -> TrackState:
    if track.gt_mid_body_spots is None:
        return TrackState.NoGT
    elif len(track.daughter_track_ids) != 1:
        return TrackState.InvalidDiv
    (correct, _, _) = track.evaluate_mid_body_detection()

    if correct:
        return TrackState.Correct
    else:
        return TrackState.WrongResult


def make_stats(
    states: dict[str, TrackState], name_filter: Optional[str] = None
) -> TrackDirStateStat:
    correct = 0
    no_gt = 0
    invalid_div = 0
    wrong = 0

    if isinstance(name_filter, str):
        filtered_states = {k: v for k, v in states.items() if name_filter in k}
    else:
        filtered_states = states

    for state in filtered_states.values():
        if state == TrackState.Correct:
            correct += 1
        elif state == TrackState.NoGT:
            no_gt += 1
        elif state == TrackState.InvalidDiv:
            invalid_div += 1
        elif state == TrackState.WrongResult:
            wrong += 1
        else:
            raise RuntimeError("Unhandled state:", state)

    return TrackDirStateStat(correct, no_gt, invalid_div, wrong)


if __name__ == "__main__":
    DEFAULT_RUN = True

    if DEFAULT_RUN:
        main()
    else:
        # Custom paths for testing
        DIR_CHOICE = 0
        DIRS = {
            0: "eval_data/Data Standard/mitoses",
            1: "eval_data/Data spastin/mitoses",
            2: "eval_data/Data cep55/mitoses",
        }

        NAME_FILTER = "converted t2_t3_F-1E5-35-8"
        main(DIRS[DIR_CHOICE], NAME_FILTER)
