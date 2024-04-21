import numpy as np

from .movie import Movie
from .movie_fmt import MovieFmt
from .movie_loading import load_movie

class Source:
    path: str
    fmt: MovieFmt

    def __init__(self, path: str, fmt: MovieFmt) -> None:
        self.path = path
        self.fmt = fmt

    def load_movie(self) -> Movie:
        return load_movie(self.path, self.fmt)
    
    def load_data(self) -> np.ndarray:
        movie = load_movie(self.path, self.fmt)
        return movie.data

    def __repr__(self) -> str:
        return f"Source(p:{self.path} fmt:{self.fmt.__repr__()})"

    def __str__(self) -> str:
        return f"{self.path}|{self.fmt}"