from .slt_debug import spatial_laptrack_debug
from wrapper_env import WrapperEnv

def spatial_laptrack_debug_runner():
    spatial_laptrack_debug(
        source           = WrapperEnv.src_file,
        slt              = WrapperEnv.main_slt_tracking_method,
        detection_method = WrapperEnv.main_spot_detection_method,
        log_fp           = WrapperEnv.log_filepath_for_purpose("slt_debug.txt"),
        out_dir          = WrapperEnv.out_dir,
        show_tracking    = False
    )