import json
from pathlib import Path
from typing import List
from dataclasses import dataclass

from .gt_point import GTPoint

@dataclass
class GTContainer:
    source_file: str
    detection_method: str
    points: List[List[GTPoint]]

    def save_to(self, filepath: str):
        output = {
            "file": self.source_file,
            "detection_method": self.detection_method,
            "max_frame_idx": len(self.points),
            "spots": {f: [{"x": p.x, "y": p.y} for p in pts] for f, pts in enumerate(self.points)}
        }

        gt_path = Path(filepath)
        gt_path.parent.mkdir(exist_ok=True, parents=True)

        print("ground truth saved to:", filepath)

        with open(filepath, "w") as file:
            json.dump(output, file, indent=2)


