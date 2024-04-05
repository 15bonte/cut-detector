import json
from typing import Any

from .bench_config import BenchConfig, DetectionConfig

def load_config(config_filepath: str) -> BenchConfig:
    dumped: dict = None
    with open(config_filepath, "r") as file:
        dumped = json.dump(file)
    assert_json(dumped, dict)

    detections = {}
    for name in dumped:
        pass

    return BenchConfig(detections)


def assert_json(v: Any, kind: type):
    if isinstance(v, kind):
        return
    else:
        raise RuntimeError(
            f"JSON config error: value {v} must be of type {kind} instead of {type(v)}"
        )
