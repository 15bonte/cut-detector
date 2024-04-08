""" Interface around the movie np array
"""

import numpy as np

class Movie:
    data: np.ndarray
    path: str
    mklp_layer: int
    sir_layer: int

    def __init__(
            self, 
            data: np.ndarray, 
            path: str, 
            mklp_layer: int = 1, 
            sir_layer: int = 0):
        self.data = data
        self.path = path
        self.mklp_layer = mklp_layer
        self.sir_layer = sir_layer

    def get_framecount(self) -> int:
        return self.movie.shape[0]
    
    def get_layercount(self) -> int:
        return self.movie.shape[3]
    
    def get_frame(self, idx: int) -> np.ndarray:
        return self.movie[idx, :, :, :].squeeze()
    
    def get_mklp_layer(self, idx: int) -> np.ndarray:
        frame = self.get_frame(idx)
        return frame[:, :, self.mklp_layer]
    
    def get_sir_layer(self, idx: int) -> np.ndarray:
        frame = self.get_frame(idx)
        return frame[:, :, self.sir_layer]
    
    def get_any_layer(self, layer_idx: int, frame_idx: int) -> np.ndarray:
        frame = self.get_frame(frame_idx)
        return frame[:, :, layer_idx]

