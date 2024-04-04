from dataclasses import dataclass
from .file_loading import MOVIE_FMT, load_movie

import numpy as np

@dataclass
class Source:
    path: str
    fmt: str

    def load(self) -> np.ndarray:
        return load_movie(self.path, self.fmt)
    
