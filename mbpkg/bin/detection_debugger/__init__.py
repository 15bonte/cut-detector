from env import StrDetectors

SRC_DIRS = [
    "src/cut_detector/data/mid_bodies_movies_test",
    "eval_data/Data cep55/movies",
    "eval_data/Data spastin/movies",
    "eval_data/Data Standard/movies",
]

DETECTORS = [
    StrDetectors.cur_log,
    StrDetectors.log2_wider,
    StrDetectors.rshift_log,
    StrDetectors.cur_dog,
]

def run_app():
    from .app import run_app
    print("--- running Detection Debugger ----")
    print("SRC_DIRS:", SRC_DIRS)
    run_app(
        SRC_DIRS,
        DETECTORS
    )