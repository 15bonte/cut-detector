
import blob_detection_bench_ws.blob_detection_bench


SANDBOX_PROJECT_LIST = """
=== Sandbox Project List ===
- hw: Hello World proof-of-concept
- d-bench: a benchmarker of detection systems
- visualizer/v/w: a blob detection and tracking visualizer.
  It is the V0.2, known previously as 'w'. Does not work
  right now because it relied upon the older playground/bpkg
  "data_loader/Movie" API
"""

SANDBOX_HELP = """
=== Sandbox Project List ===
Sandbox projects are an intermediate between a library and
a binary.
They are neither meant to be controlled through API like libraries,
nor to be "simple executables" without configurations (like the
library runners).

They are meant to be modified directly.
"""
__doc__ = SANDBOX_HELP

def run_sandbox_project(name: str):

    if name == "list":
        print(SANDBOX_PROJECT_LIST)
    elif name == "help":
        print(SANDBOX_HELP)
    elif name == "hw":
        from .hw import hello_world
        hello_world()
    elif name == "d-bench":
        from blob_detection_bench_ws import run_detec_bench
        run_detec_bench()
    elif name in ["visualizer", "v", "w"]:
        from visualizer import run_app
        run_app()
    elif name == "nls-example":
        from mb_layers.example import run_example
        run_example()
    elif name == "multi-ppl":
        from multi_pipeline import run_multi_pipeline
        run_multi_pipeline()
    else:
        print(f"Sandbox '{name}' does not exist")

