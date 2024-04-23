from mbpkg.movie_loading import Source

from env import SourceFiles, get_associated_gt_path

SOURCE = SourceFiles.example
GT_FP = get_associated_gt_path(SOURCE)

def run_app():
    from .app import start_app
    print("== starting ground truth editor ===")
    print("source:", SOURCE)
    print("GT_FP:", GT_FP)
    start_app(SOURCE, GT_FP)