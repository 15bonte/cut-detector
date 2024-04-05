import json

from .bench_config import BenchConfig, DetectionConfig

def write_config(config: BenchConfig, to_config_filepath: str):
    bench_d = {}
    for name, conf in config.detections.items():
        conf_d = {"kind": conf.kind}
        for arg, v in conf.args.items():
            conf_d[arg] = v
        bench_d[name] = conf_d
    
    with open(to_config_filepath, "w") as file:
        json.dump(bench_d, file, indent=2)
