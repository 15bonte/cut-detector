from cnn_framework.utils.readers.tiff_reader import TiffReader

import numpy as np

from .movie import Movie
from .movie_fmt import MovieFmt

def load_movie(path: str, fmt: MovieFmt) -> np.ndarray:
    if fmt == MovieFmt.four_channel:
        data = load_std_movie_data(path)
    elif fmt == MovieFmt.three_channel:
        data = load_long3c_movie_data(path)
    else:
        raise RuntimeError(f"Could not load movie '{path}': Unsupported kind '{fmt}'")
    
    return Movie(data, path)


def load_std_movie_data(path) -> np.ndarray:
    image = TiffReader(path, respect_initial_type=True).image  # TCZYX
    movie = image[:, :3, ...].squeeze()  # T C=3 YX
    movie = movie.transpose(0, 2, 3, 1)  # TYXC
    return movie # -> TYXC


def load_long3c_movie_data(path) -> np.ndarray:
    image = TiffReader(path, respect_initial_type=True).image  # ZCTYX
    movie = image[:, :3, ...].squeeze()  # Z C=3 TYX -> C=3 TYX
    movie = movie.transpose(1, 2, 3, 0)  # CTYX -(1,2,3,0)> TYXC
    return movie # -> TYXC
