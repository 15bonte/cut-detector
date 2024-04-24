import os
import pickle

from cut_detector.utils.mitosis_track import MitosisTrack

SOURCE_CHOICE = 2
SOURCES = {
    0: "eval_data/Data Standard/mitoses",
    1: "eval_data/Data spastin/mitoses",
    2: "eval_data/Data cep55/mitoses",
}

def main():
    print(f"WARNING: you are about to reset all bin files in:")
    print(SOURCES[SOURCE_CHOICE])
    print("while deleting data is almost instantaneous, regenerating them takes some time (30+min)")
    print("")
    print("enter 'YES' to continue. Anything else will cancel the operation")
    print("")

    ans = input("[Proceed ?] >> ")
    if ans != "YES":
        print("cancelling the operation")
        return
    
    print("resetting:", SOURCES[SOURCE_CHOICE])
    with os.scandir(SOURCES[SOURCE_CHOICE]) as it:
        es = [e for e in it if e.is_file(follow_symlinks=False) and e.name.endswith(".bin")]
    for e in es:
        with open(e.path, "rb") as f:
            track: MitosisTrack = pickle.load(f)
        track.mid_body_spots = {}
        with open(e.path, "wb") as f:
            pickle.dump(track, f)
        print("-", e.path)


if __name__ == "__main__":
    main()