""" Entry file when invocking bpkg as a binary
"""

import sys

BPKG_HELP = """
=== Blob Package (bpkg) binary entry ===

bpkg is both a library package and a 'toolbox'
that wraps several executables.
To use it as a library, just import the submodules
directly, ie
'from bpkg.importation import load_movie'

As an executable, syntax is the following:
'python bpkg [COMMAND]'

Commands are executed with arguments that are hardcoded
in the modules.
Usually each module has 'main' (not called main) functions
(this is indicated in the documentation) and
runner functions, that bear the same name as their
respective main function but with the suffix '_runner'.
These wrapper functions use constants that are defined
in the '_runner' modules.


The following commands are available:

gen-d-gt:
    Generate a '''ground truth''' file from a file with the 
    given detection pipeline.

bench-d-gt:
    Bench a pipeline on a source and its associated ground
    truth.

new-d-gt:
    Opens an application that allows one to write
    a ground truth file

slt-debug:
    Run a debug version of Spatial LapTrack, with a log file 
    and resulting images

gen-dbench-conf:
    Generate a new detection Bench Configuration file, with all
    current detectors

run-dbench:
    run the detection benching on the configuration pipelines,
    using associated ground truth files

d-debug <mode>:
    Starts the detection debugging tool. Mode can be one of the following:
      - cli

sand <project>:
    starts the sandbox project <project>. To get a list of all sandbox
    projects, use 'sand list'. If you don't know what the
    sandbox projects are, use 'sand help'

"""
    
def print_help():
    print(BPKG_HELP)

def run_slt_debug():
    from slt_debug.slt_debug_runner import spatial_laptrack_debug_runner
    spatial_laptrack_debug_runner()

def run_gen_d_gt():
    from detection_gt.generation.generation_runner import generate_ground_truth_runner
    generate_ground_truth_runner()

def run_bench_d_gt():
    from detection_gt.bench.bench_runner import bench_detection_against_gt_runner
    bench_detection_against_gt_runner()

def run_new_d_gt():
    from detection_gt.app.runner import start_app_runner
    start_app_runner()

def run_slt_debug():
    from slt_debug.slt_debug_runner import spatial_laptrack_debug_runner
    spatial_laptrack_debug_runner()

def run_d_debug():
    from detection_debug import run_app
    arg = sys.argv[2] if len(sys.argv) >= 3 else None
    run_app(arg)

def run_gen_dbench_conf():
    from detection_bench.runner.gen_config_runner import gen_config_runner
    gen_config_runner()

def run_dbench():
    from detection_bench.runner.exec_config_runner import exec_runner
    exec_runner()

def run_sand():
    arg = sys.argv[2]
    from sandbox import run_sandbox_project
    run_sandbox_project(arg)

FLAG_TABLE = {
    "help":            print_help,
    "--help":          print_help,
    "-h":              print_help,
    "h":               print_help,
    "slt-debug":       run_slt_debug,
    "gen-d-gt":        run_gen_d_gt,
    "bench-d-gt":      run_bench_d_gt,
    "new-d-gt":        run_new_d_gt,
    "slt-debug":       run_slt_debug,
    "d-debug":         run_d_debug,
    "gen-dbench-conf": run_gen_dbench_conf,
    "run-dbench":      run_dbench,
    "d-debug":         run_d_debug,
    "sand":            run_sand,
}

def run_bpkg():
    argc = len(sys.argv)
    if argc == 1:
        print(BPKG_HELP)
    else:
        cmd = sys.argv[1]
        fn = FLAG_TABLE.get(cmd, None)
        if fn is None:
            print(f"Unknown command {cmd}, see help.")
        else:
            fn()
