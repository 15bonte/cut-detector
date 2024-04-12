import json
from typing import Any

from .bench_config import BenchConfig, DetectionConfig

def load_config(config_filepath: str) -> BenchConfig:
    dumped: dict = None
    with open(config_filepath, "r") as file:
        dumped = json.load(file)
    assert_config_type(dumped, dict)

    detections: dict[str, DetectionConfig] = {}

    detection_name: str
    detection_settings: dict
    for detection_name, detection_settings in dumped.items():
        assert_config_type(detection_settings, dict)
        kind: str = assert_config_type(detection_settings.pop("kind"), str)
        if not kind in ["log", "dog", "doh"]:
            raise RuntimeError(f"Unsupported detection type {kind}")
        detections[detection_name] = DetectionConfig(kind, detection_settings)

    return BenchConfig(detections)


def assert_config_type(v: Any, kind: type) -> Any:
    if isinstance(v, kind):
        return v
    else:
        raise RuntimeError(
            f"Detection Config Error: value {v}, has type {type(v)} instead of expected {kind}"
        )