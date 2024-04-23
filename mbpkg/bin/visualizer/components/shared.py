""" Shared globals and their handling
"""

import os.path
import json

from mbpkg.movie_loading import Movie, Source, MovieFmt
from mbpkg.better_detector import Detector

### Set at start ###
_mitosis_src_dirpaths: list[str] = []
_detectors: list[Detector] = []

### Set during execution ###
_movie: Movie = None



#### Functions ####
def init_mitosis_src_dirpaths(p: list[str]):
    global _mitosis_src_dirpaths
    _mitosis_src_dirpaths = p

def init_detectors(d: list[Detector]):
    global _detectors
    _detectors = d


def sig_load_movie(dirpath: str, filename: str, fmt: str) -> str:
    """ Loads the movie and returns the movie loaded signal"""
    global _movie
    filepath = os.path.join(dirpath, filename)
    _movie = Source(
        filepath,
        MovieFmt(fmt)
    ).load_movie()

    print("movie imported")

    return json.dumps({
        "path": filepath,
        "fmt": fmt,
    })


