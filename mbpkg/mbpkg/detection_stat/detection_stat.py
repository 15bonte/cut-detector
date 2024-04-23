import json
from dataclasses import dataclass
from typing import Optional

import numpy as np

from mbpkg.helpers.logging import manage_logger, Logger
from mbpkg.better_detector import Detector


@dataclass
class DetectionStat:
    detector: Detector
    source_path: str
    distances: np.ndarray
    time: Optional[float] = None
    fn_count: int = 0
    fp_count: int = 0


    def compute_base_stats(self) -> dict[str, float]:
        m   = np.min(self.distances)
        p5  = np.percentile(self.distances, 5)
        q1  = np.percentile(self.distances, 25)
        med = np.median(self.distances)
        q3  = np.percentile(self.distances, 75)
        p95 = np.percentile(self.distances, 95)
        M   = np.max(self.distances)

        return {
            "min": m,
            "p5": p5,
            "q1": q1,
            "med": med,
            "q3": q3,
            "p95": p95,
            "max": M
        }


    def report(self, filepath: Optional[str], should_print: bool = True):
        with manage_logger(filepath, should_print) as l:
            self.report_with_logger(l)


    def report_with_logger(self, l: Logger):
        l.log("=====")
        l.log("detector:", self.detector.to_str())
        l.log("source:", self.source_path)
        l.log("--")
        l.log("time:", "NA" if self.time is None else self.time)
        l.log("missed:", self.fn_count)
        l.log("created:", self.fp_count)
        l.log(self.compute_base_stats())
        l.log("=====")


    def write_stat(self, path: str):
        d = {
            "detector": self.detector.to_str(),
            "source_path": self.source_path,
            "fn_count": self.fn_count,
            "fp_count": self.fp_count,
            "distances:": json.dumps(self.distances)
        }
        if self.time is not None:
            d["time"] = self.time

        with open(path, "w") as f:
            json.dump(d, f, indent=2)
    


