import json

import numpy as np

from mbpkg.helpers.logging import manage_logger
from mbpkg.better_detector import Detector
from mbpkg.detection_stat import DetectionStat

class MultiBenchStat:
    source_path: str
    individual_benches: dict[Detector, DetectionStat]

    def __init__(
            self, 
            source_path: str,
            individual_benches: dict[Detector, DetectionStat]
            ) -> None:
        self.source_path = source_path
        self.individual_benches = individual_benches


    def write_result(self, filepath: str):
        d = {
            "source_path": self.source_path,
            "distances": {k.to_str(): b.distances.tolist() for k, b in self.individual_benches.items()},
            "fn_counts": {k.to_str(): b.fn_count for k, b in self.individual_benches.items()},
            "fp_counts": {k.to_str(): b.fp_count for k, b in self.individual_benches.items()},
            "times":     {k.to_str(): b.time for k, b in self.individual_benches.items()},
        }
        with open(filepath, "w") as file:
            json.dump(d, file, indent=2)


    def report(self, log_filepath: str, should_print: bool):
        with manage_logger(log_filepath, should_print) as l:
            l.log("######## Multi-Detection Bench ########")
            l.log("source:", self.source_path)
            for d_stat in list(self.individual_benches.values()):
                l.log("")
                d_stat.report_with_logger(l)


    



