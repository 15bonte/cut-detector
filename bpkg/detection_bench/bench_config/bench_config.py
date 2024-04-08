from typing import Dict, Any, Literal, Callable
from functools import partial

import numpy as np

from cut_detector.factories.mb_support import detection

class DetectionConfig:
    DetectionKind = Literal["log", "dog", "doh"]

    kind: DetectionKind
    args: Dict[str, Any]

    def __init__(self, kind: DetectionKind, args: Dict[str, Any]):
        if not kind in ["log", "dog", "doh"]:
            raise RuntimeError(f"DetectionConfig does not recognize kind {kind}")
        self.kind = kind
        self.args = args

    def make_associated_detector(self) -> Callable[[np.ndarray], np.ndarray]:
        """Makes the detector that would correspond to this
        configuration"""
        mapping = {
            "log": detection.detect_minmax_log,
            "dog": detection.detect_minmax_dog,
            "doh": detection.detect_minmax_doh
        }
        return partial(
            mapping[self.kind],
            **self.args
        )

class BenchConfig:
    detections: Dict[str, DetectionConfig]

    def __init__(self, detections: Dict[str, DetectionConfig]):
        self.detections = detections