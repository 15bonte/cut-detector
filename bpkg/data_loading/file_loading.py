from typing import Literal
from cnn_framework.utils.readers.tiff_reader import TiffReader
import numpy as np


MOVIE_FMT = Literal["4c", "3c"]


def load_movie(path: str, fmt: MOVIE_FMT) -> np.ndarray:
    print(f"loading {path} | {fmt}")
    if fmt == "4c":
        return load_std_movie(path)
    elif fmt == "3c":
        return load_long3c_movie(path)
    else:
        raise RuntimeError(f"Could not load movie '{path}': Unsupported kind '{fmt}'")


def load_std_movie(path) -> np.ndarray:
    image = TiffReader(path, respect_initial_type=True).image  # TCZYX
    movie = image[:, :3, ...].squeeze()  # T C=3 YX
    movie = movie.transpose(0, 2, 3, 1)  # TYXC
    return movie # -> TYXC


def load_long3c_movie(path) -> np.ndarray:
    image = TiffReader(path, respect_initial_type=True).image  # ZCTYX
    movie = image[:, :3, ...].squeeze()  # Z C=3 TYX -> C=3 TYX
    movie = movie.transpose(1, 2, 3, 0)  # CTYX -(1,2,3,0)> TYXC
    return movie # -> TYXC
