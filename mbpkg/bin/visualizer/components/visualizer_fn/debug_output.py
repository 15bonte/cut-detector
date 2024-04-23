from typing import Any, Union
from dataclasses import dataclass

@dataclass
class SigmaLayoutOutput:
    sigma: float

DebugOutput = Union[SigmaLayoutOutput]

class DebugOutputError(Exception):
    pass

def make_debug_output(d: dict[str, Any]) -> DebugOutput:
    if (layer := d.get("layer")) is None:
        raise DebugOutputError(f"Missing required key 'layer' in DebugOutput dict: {d}")

    if layer == "sigma_layer":
        if (s := d.get("viz_sigma")) is None:
            raise DebugOutputError(f"Missing required key 'vis_sigma' in DebugOutput dict: {d}")
        return SigmaLayoutOutput(s)
    
    else:
        raise DebugOutputError(f"Unsupported layer {layer} in dict {d}")


