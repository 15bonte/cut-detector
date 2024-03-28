import numpy as np

from .load_movie import load_movie, MOVIE_KIND_LITERAL

class Movie:
    movie: np.array
    mklp_layer: int
    sir_layer: int

    def __init__(
            self,
            path: str,
            kind: MOVIE_KIND_LITERAL,
            mklp_layer: int = 1,
            sir_layer: int = 0):
        self.movie = load_movie(path, kind) # TYXC
        self.mklp_layer = mklp_layer
        self.sir_layer = sir_layer

    @classmethod
    def make_movie_from_array(
        cls,
        data: np.array,
        mklp_layer: int = 1,
        sir_layer: int = 0):
        """ Array must be well-formatted
        """
        
        m = cls.__new__(cls) # empty unitialized instance
        m.movie = data
        m.mklp_layer = mklp_layer
        m.sir_layer = sir_layer
        return m 

    def get_framecount(self) -> int:
        return self.movie.shape[0]
    
    def get_layercount(self) -> int:
        return self.movie.shape[3]
    
    def get_frame(self, idx: int) -> np.array:
        return self.movie[idx, :, :, :].squeeze()
    
    def get_mklp_frame(self, idx: int) -> np.array:
        frame = self.get_frame(idx)
        return frame[:, :, self.mklp_layer]
    
    def get_sir_frame(self, idx: int) -> np.array:
        frame = self.get_frame(idx)
        return frame[:, :, self.sir_layer]
    
    def get_layer_frame(self, frame_idx: int, layer_idx: int) -> np.array:
        frame = self.get_frame(frame_idx)
        return frame[:, :, layer_idx]


