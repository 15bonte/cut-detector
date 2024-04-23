""" Creating files in out dir, or getting the path to this dir
"""

from os.path import join

OUT_DIR = "src/cut_detector/data/mid_bodies_movies_test"

def make_log_file(name: str) -> str:
    return join(OUT_DIR, name)


