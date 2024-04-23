""" Main module is where one can set the constants use for app invocation
"""

from env.better_detectors import Detectors

SRC_DIRPATHS = [
    "src/cut_detector/data/mid_bodies_movies_test",
    "eval_data/Data Standard/movies",
    "eval_data/Data spastin/movies",
    "eval_data/Data cep55/movies",
]

DETECTORS = [
    Detectors.lapgau,
    Detectors.log2_wider,
    Detectors.rshift_log,

    Detectors.diffgau,

    Detectors.hessian,
]

def drive_app():
    from .app import start_app
    start_app(
        mitosis_src_dirpaths=SRC_DIRPATHS,
        detectors=DETECTORS,
    )