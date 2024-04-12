from wrapper_env import WrapperEnv
from .app import start_app

def start_app_runner():
    mv_filepath = WrapperEnv.src_file
    gt_filepath = WrapperEnv.gt_filepath_for_source(mv_filepath)
    start_app(mv_filepath, gt_filepath)