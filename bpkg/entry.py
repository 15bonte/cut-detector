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

"""
    
def print_help():
    print(BPKG_HELP)

def run_gen_d_gt():
    from detection_gt.generation.runner import generate_ground_truth_runner
    generate_ground_truth_runner()

def run_bench_d_gt():
    from detection_gt.bench.runner import bench_detection_against_gt_runner
    bench_detection_against_gt_runner()

def run_new_d_gt():
    from detection_gt.app.runner import start_app_runner
    start_app_runner()

FLAG_TABLE = {
    "help":       print_help,
    "--help":     print_help,
    "-h":         print_help,
    "h":          print_help,
    "gen-d-gt":   run_gen_d_gt,
    "bench-d-gt": run_bench_d_gt,
    "new-d-gt":   run_new_d_gt,
}

def run_bpkg():
    argc = len(sys.argv)
    if argc == 1:
        print(BPKG_HELP)
    else:
        cmd = sys.argv[1]
        fn = FLAG_TABLE.get(sys.argv[1], None)
        if fn is None:
            print(f"Unknown command {cmd}, see help.")
        else:
            fn()
