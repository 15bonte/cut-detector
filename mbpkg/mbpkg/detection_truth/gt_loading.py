import json
from typing import Optional

from mbpkg.helpers.json_processing import extract, assert_extraction, ExtractionError

from .detection_gt import DetectionGT, DetectionSpot
from .detection_gt_fmt_error import DetectionGtFmtError

def load_gt(path: str):
    """ 
    Loads ground truth from the specified file

    raises FileNotFoundError if the file could not be found,
    and DetectionGTFmtError if the format is not respected (missing keys,
    wrong types, etc...)
    """
    with open(path, "r") as file:
        data = json.load(file)

    try:
        version: Optional[str]  = extract(data, "version", str, allow_missing=True)
        src_file: str           = extract(data, "file", str)
        method: str             = extract(data, "detection_method", str)
        spots: dict             = extract(data, "spots", dict)
        spot_dict               = {}
        for f_idx_str, spots in spots.items():
            f_idx = int(f_idx_str)
            assert_extraction(spots, list)

            points = []
            for spot in spots:
                assert_extraction(spot, dict)
                points.append(DetectionSpot(
                    spot["x"],
                    spot["y"]
                ))
            if len(points) > 0:
                spot_dict[f_idx] = points
            
    except (ExtractionError, ValueError, KeyError) as e:
        # ValueError for int(f_idx_str)
        # KeyError   for spot["x"/"y"]
        raise DetectionGtFmtError(e.args) from e
    
    return DetectionGT(
        file,
        method,
        spot_dict
    )

