from wrapper_env import WrapperEnv
from ..bench_config import generate_config, write_config

def gen_config_runner():
    conf = generate_config()
    write_config(
        conf,
        WrapperEnv.get_config_file_path("detectors.json")
    )