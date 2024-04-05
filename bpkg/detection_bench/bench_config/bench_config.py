from typing import Dict, Any

class DetectionConfig:
    kind: str
    args: Dict[str, Any]

    def __init__(self, kind: str, args: Dict[str, Any]):
        self.kind = kind
        self.args = args


class BenchConfig:
    detections: Dict[str, DetectionConfig]

    def __init__(self, detections: Dict[str, DetectionConfig]):
        self.detections = detections