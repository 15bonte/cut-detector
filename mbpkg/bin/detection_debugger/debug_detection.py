from typing import Literal, Any

import numpy as np

from mbpkg.detector import Detector
from .debug_funcs import debug_log, debug_dog, debug_doh

def debug_detector(
        image: np.ndarray,
        base_detector_repr: str, 
        override_params: dict[str, Any], 
        output: Literal["s_layer", "s_cube"],
        ) -> dict:


    base_detector_kind = Detector(base_detector_repr).get_spot_method_kind()

    m = {
        Detector.SpotMethodKind.log: debug_log,
        Detector.SpotMethodKind.dog: debug_dog,
        Detector.SpotMethodKind.doh: debug_doh,
    }

    if (debug_func := m.get(base_detector_kind)) is not None:
        debug_func(image, output, override_params)
    else:
        return {}
