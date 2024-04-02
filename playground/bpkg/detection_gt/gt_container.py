from typing import List
from dataclasses import dataclass

@dataclass
class GTPoint:
    x: int
    y: int

@dataclass
class GTContainer:
    source_file: str
    detection_mode: str
    points: List[List[GTPoint]]