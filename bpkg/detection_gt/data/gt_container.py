from typing import List
from dataclasses import dataclass

from .gt_point import GTPoint

@dataclass
class GTContainer:
    source_file: str
    detection_method: str
    points: List[List[GTPoint]]