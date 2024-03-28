import numpy as np

from typing import Literal
from cnn_framework.utils.readers.tiff_reader import TiffReader

MOVIE_KIND_LITERAL = Literal["4c", "3c"]

def load_movie(path: str, kind: MOVIE_KIND_LITERAL) -> np.array:
    print(f"loading {path} | {kind}")
    if kind == "4c":
        return load_std_movie(path)
    elif kind == "3c":
        return load_long_3channels(path)
    else:
        raise RuntimeError("Unsupported kind:", MOVIE_KIND_LITERAL)
    
def load_std_movie(path: str) -> np.array:
    image = TiffReader(path, respect_initial_type=True).image  # TCZYX
    print("image shape:", image.shape)
    movie = image[:, :3, ...].squeeze()  # T C=3 YX
    movie = movie.transpose(0, 2, 3, 1)  # TYXC
    return movie # -> TYXC

def load_long_3channels(path: str) -> np.array:
    image = TiffReader(path, respect_initial_type=True).image  # ZCTYX
    print("image shape:", image.shape)
    movie = image[:, :3, ...].squeeze()  # Z C=3 TYX -> C=3 TYX
    movie = movie.transpose(1, 2, 3, 0)  # CTYX -(1,2,3,0)> TYXC
    return movie # -> TYXC
