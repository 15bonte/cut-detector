from dataclasses import dataclass
from typing import Optional

from data_loading import MOVIE_FMT

from .common import get_list


@dataclass
class LoadSource:
    path: str
    fmt: MOVIE_FMT


def parse_load_source(parts: list[str]) -> Optional[LoadSource]:
    if len(parts) <= 1:
        print("load source must take at least the path to source")
        return None
    
    path = parts[1]

    fmt: str = get_list(parts, 2, "4c")
    if fmt == "":
        fmt = "4c"
    if not fmt in ["4c", "3c"]:
        print(f"format, if provided, must be either 4c or 3c, found '{fmt}'")
        return None
    
    return LoadSource(path, fmt)
    