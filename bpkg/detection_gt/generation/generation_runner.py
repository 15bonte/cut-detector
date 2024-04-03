from wrapper_env import WrapperEnv
from .generation import generate_ground_truth

def generate_ground_truth_runner():
    gt_filepath = WrapperEnv.gt_filepath_from_source(WrapperEnv.src_file)
    print("##### at start of runner: gt_filepath:", gt_filepath)
    generate_ground_truth(
        s=WrapperEnv.src_file,
        detection_method=WrapperEnv.main_spot_detection_method,
        gt_filepath=gt_filepath
    )