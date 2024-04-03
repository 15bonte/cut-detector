from functools import partial
from .detection_func import detect_minmax_log, detect_minmax_dog, detect_minmax_doh

current_log = partial(
    detect_minmax_log, 
    min_sigma=5,
    max_sigma=10,
    num_sigma=5,
    threshold=0.1
)

current_dog = partial(
    detect_minmax_dog,
    min_sigma=2,
    max_sigma=5,
    sigma_ration=1.2,
    threshold=0.1,
)

current_doh = partial(
    detect_minmax_dog,
    min_sigma=5,
    max_sigma=10,
    num_sigma=5,
    threshold=0.0040
)