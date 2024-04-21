from typing import Literal

from .movie_fmt_error import MovieFmtError

class MovieFmt:
    fmt: int

    four_channel  = 0
    three_channel = 1

    FMT_STR = Literal[
        "4c",
        "3c"
    ]

    def __init__(self, format: FMT_STR):
        if format == "4c":
            self.fmt = MovieFmt.four_channel
        elif format == "3c":
            self.fmt = MovieFmt.three_channel
        else:
            raise MovieFmtError(format, ["4c", "3c"])
        
    @staticmethod
    def available_fmt_strs() -> list[str]:
        return ["4c", "3c"]


    def __eq__(self, value: object) -> bool:
        if isinstance(value, MovieFmt):
            return self.fmt == value.fmt
        elif isinstance(value, int):
            return self.fmt == value
        elif isinstance(value, str):
            return self.__str__() == value
        else:
            return NotImplemented


    def __str__(self) -> str:
        if self.fmt == MovieFmt.four_channel:
            return "4c"
        elif self.fmt == MovieFmt.three_channel:
            return "3c"
        else:
            raise RuntimeError(f"MovieFmt is in an invalid state: {self.fmt}")


    def __repr__(self) -> str:
        if self.fmt == MovieFmt.four_channel:
            return "MovieFmt.4c"
        elif self.fmt == MovieFmt.three_channel:
            return "MovieFmt.3c"
        else:
            raise RuntimeError(f"MovieFmt is in an invalid state: {self.fmt}")