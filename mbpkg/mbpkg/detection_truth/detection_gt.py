import json
from dataclasses import dataclass


@dataclass
class DetectionSpot:
    x: int
    y: int


class DetectionGT:
    file: str
    detection_method: str
    spot_dict: dict[int, list[DetectionSpot]]

    def __init__(self, file: str, method: str, spot_dict: dict[int, list[DetectionSpot]]):
        self.file = file
        self.detection_method = method
        self.spot_dict = spot_dict

    def write_gt(self, path: str):
        with open(path, "w") as file:
            json.dump(self, file, indent=2)

