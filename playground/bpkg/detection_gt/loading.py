import json
from typing import Any
from .gt_container import GTContainer, GTPoint

def load_gt(fp: str) -> GTContainer:
    with open(fp, "r") as file:
        d = json.load(file)
        # print(f"d:\n{d}")
        source_fp: str = assert_type(d["file"], str)
        detection_mode: str = assert_type(d["detection_mode"], str)
        max_frame_idx: int = assert_type(d["max_frame_idx"], int)
        spots: dict = assert_type(d["spots"], dict)
        spot_list = [[] for _ in range(max_frame_idx+1)]
        for frame in spots:
            l = []
            for spot in spots[frame]:
                x: int = assert_type(spot["x"], int)
                y: int = assert_type(spot["y"], int)
                l.append(GTPoint(x, y))
            spot_list[int(frame)] = l
        
        c = GTContainer(source_fp, detection_mode, spot_list)
        # print(c.__dict__)
        return c

def assert_type(v: Any, kind: type) -> Any:
    if isinstance(v, kind):
        return v
    else:
        raise RuntimeError(f"Invalid kind: value is {type(v)} instead of {kind}\nValue: {v}")