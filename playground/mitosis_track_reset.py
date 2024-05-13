import os
import pickle

from cut_detector.utils.mitosis_track import MitosisTrack


def main(mitoses_path: str):
    """Playground function to reset all mid-body bin files in a folder.
    BE VERY CAREFUL - no default since it is not supposed to reset default data.
    """
    print("WARNING: you are about to reset all bin files in:")
    print(mitoses_path)
    print(
        "while deleting data is almost instantaneous, regenerating them takes some time (30+min)"
    )
    print("")
    print("enter 'YES' to continue. Anything else will cancel the operation")
    print("")

    ans = input("[Proceed ?] >> ")
    if ans != "YES":
        print("cancelling the operation")
        return

    print("resetting:", mitoses_path)
    with os.scandir(mitoses_path) as it:
        es = [
            e
            for e in it
            if e.is_file(follow_symlinks=False) and e.name.endswith(".bin")
        ]
    for e in es:
        with open(e.path, "rb") as f:
            track: MitosisTrack = pickle.load(f)
        track.mid_body_spots = {}
        with open(e.path, "wb") as f:
            pickle.dump(track, f)
        print("-", e.path)


if __name__ == "__main__":
    # Only custom paths
    SOURCE_CHOICE = 2
    SOURCES = {
        0: "eval_data/Data Standard/mitoses",
        1: "eval_data/Data spastin/mitoses",
        2: "eval_data/Data cep55/mitoses",
    }
    main(SOURCES[SOURCE_CHOICE])
