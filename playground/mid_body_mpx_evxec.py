""" Multiplexing the evaluation/execution process
"""
import sys
import subprocess
from os.path import join
from pathlib import Path
from typing import Callable, Union
from cut_detector.factories.mb_support import detection, tracking

from mid_body_execution import start_execution
from mid_body_evaluation import start_evaluation

DETECTORS = [
    detection.cur_log,
    detection.cur_dog,
    detection.cur_doh,
    "h_maxima",
]
SOURCES = [
    "eval_data/Data Standard",
    "eval_data/Data spastin",
    "eval_data/Data cep55",
]
REPORT_DIR = "playground/mpx"
PARALLELIZE = True

def start_evexc(
        sources: list[str], 
        detectors: list[Union[Callable, str]],
        report_dir: str,
        parallelize_detection: bool):
    

    raise RuntimeError("Not working yet")
    
    # 0 is path to this file; 
    # true args start at 1
    args = sys.argv 
    if len(sys.argv) == 1:
        # start sub processes
        print("main process")
        for src_count in range(len(SOURCES)):
            subprocess.run(["python", args[0], str(src_count)])
    else:
        # run associated pipeline
        print("subprocess")
        src_id = int(args[1])
        detector_timing: list[list[float]] = []
        detector_eval: list[dict] = []
        for detector in detectors:
            detector_timing.append(start_execution(
                data_set_path=SOURCES[src_id],
                mid_body_detection_method=detector,
                mid_body_tracking_method=tracking.cur_spatial_laptrack,
                parallel_detection=parallelize_detection
            ))
            detector_eval.append(start_evaluation(
                mitoses_folder = join(SOURCES[src_id], "mitoses"),
            ))
        
        original_stdout = sys.stdout
        for (timing, perf, detector_idx) in zip(detector_timing, detector_eval, range(len(detectors))):
            src_p = Path(SOURCES[src_id])
            
            with open(join(report_dir, src_p.stem, ".txt"), "w") as file:
                
                print("path:", SOURCES[src_id])
                print("method index:", detector_idx)
                print("")
                print("timing:")
                for t_idx, t in enumerate(timing):
                    print(f"- {t_idx}/{len(timing)} video: {t:.3f}s")
                print("")
                print("performance:")
                print_eval_report(perf)
                print("\n\n\n")
                
        sys.stdout = original_stdout

def print_eval_report(eval_dict: dict):
    {
        "wrong_detections": wrong_detections,
        "mb_detected": mb_detected,
        "mb_not_detected": mb_not_detected
    }
    wrong_detections = eval_dict["wrong_detections"]
    mb_detected      = eval_dict["mb_detected"]
    mb_not_detected  = eval_dict["mb_not_detected"]

    if len(wrong_detections) > 0:
        print("\nWrong detections:")
        for wrong_detection in wrong_detections:
            print(
                f"{wrong_detection['path']}: detected {wrong_detection['percent_detected']}% with avg distance {wrong_detection['average_position_difference']}"
            )

    if (mb_detected + mb_not_detected) == 0:
        print("No mid-body detection evaluation possible.")
    else:
        print(
            f"\nMid-body detection evaluation: {mb_detected / (mb_detected + mb_not_detected) * 100:.2f}% | {mb_detected}/{mb_detected + mb_not_detected}"
        )

if __name__ == "__main__":
    start_evexc(
        DETECTORS,
        SOURCES,
        REPORT_DIR,
        PARALLELIZE
    )