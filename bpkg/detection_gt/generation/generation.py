import json
from pathlib import Path
from cut_detector.factories.mid_body_detection_factory import MidBodyDetectionFactory
from data_loading import Source

def generate_ground_truth(
        s: Source, 
        detection_method: MidBodyDetectionFactory.SPOT_DETECTION_MODE,
        gt_filepath: str):
    print("gt filepath:", gt_filepath)
    print("generating Ground Truth for", s.path, "using", detection_method)
    movie_data = s.load()
    factory = MidBodyDetectionFactory()
    spots_dict = factory.detect_mid_body_spots(movie_data, mode=detection_method)
    spots_idx_list = list(spots_dict.keys())
    spots_idx_list.sort()
    max_frame_idx = spots_idx_list[-1]
    output = {
        "file": s.path,
        "detection_method": detection_method,
        "max_frame_idx": max_frame_idx,
        "spots": {k: [{"x": v.x, "y": v.y} for v in spots_dict[k]] for k in spots_dict}
    }

    print("writing ground truth to:", gt_filepath)
    gt_path = Path(gt_filepath)
    gt_path.parent.mkdir(exist_ok=True, parents=True)
    with open(gt_filepath, "w") as file:
        json.dump(output, file, indent=2)