import json
from typing import Any
from .gt_container import GTContainer
from .gt_point import GTPoint

def load_gt_file(filepath: str) -> GTContainer:
    with open(filepath, "r") as file:
        d = json.load(file)
        source_fp: str = assert_type(d["file"], str)
        detection_method: str = assert_type(d["detection_method"], str)
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
        
        c = GTContainer(source_fp, detection_method, spot_list)
        return c


def assert_type(v: Any, kind: type) -> Any:
    if isinstance(v, kind):
        return v
    else:
        raise RuntimeError(f"Invalid kind: value is {type(v)} instead of {kind}\nValue: {v}")


