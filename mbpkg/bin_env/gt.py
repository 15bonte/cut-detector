from os.path import join
from pathlib import Path

from mbpkg.movie_loading import Source

GT_DIR = "src/cut_detector/data/mid_bodies_movies_test/gt"

def get_associated_gt_path(src: Source) -> str:
    src_path = Path(src.path)
    stem = join(GT_DIR, src_path.stem)
    return f"{stem}__gt.json"


