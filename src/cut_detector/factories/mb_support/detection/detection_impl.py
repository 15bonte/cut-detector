from functools import partial
from .detection_func import detect_minmax_log, detect_minmax_dog, detect_minmax_doh


############ Laplacian of Gaussian ############
lapgau = partial(
    detect_minmax_log, 
    min_sigma=5,
    max_sigma=10,
    num_sigma=5,
    threshold=0.1
)

log2_wider = partial(
    detect_minmax_log, 
    min_sigma=2,
    max_sigma=8,
    num_sigma=4,
    threshold=0.1
)

rshift_log = partial(
    detect_minmax_log, 
    min_sigma=3,
    max_sigma=11,
    num_sigma=5,
    threshold=0.1
)

############ Difference of Gaussian ############

diffgau = partial(
    detect_minmax_dog,
    min_sigma=2,
    max_sigma=5,
    sigma_ration=1.2,
    threshold=0.1,
)

############ Determinant of Hessian ############

hessian = partial(
    detect_minmax_dog,
    min_sigma=5,
    max_sigma=10,
    num_sigma=5,
    threshold=0.0040
)

