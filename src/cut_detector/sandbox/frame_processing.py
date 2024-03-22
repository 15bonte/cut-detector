""" Allows one to process a movie frame by frame
"""

import numpy as np

from typing import Callable, Any

def process_movie(movie: np.array, filename: str, processing_fun: Callable[[np.array, str, int], Any]):
    """ Runs a processing function on a movie frame [YXC], with all channels
    as well as filename and frame index
    """
    print("processing movie:", filename)
    n_frames = movie.shape[0]
    for frame in range(n_frames):
        print(f"processing frame ({frame}/{n_frames-1})")
        mitosis_frame = movie[frame, :, :, :]
        processing_fun(mitosis_frame, filename, frame)
