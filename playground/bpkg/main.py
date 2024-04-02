""" Common entry for all "next" binaries and libraries
"""

import sys
from textwrap import dedent

TEST_FILE_DIR = "./src/cut_detector/data/mid_bodies_movies_test"
TEST_FILE_OUT_DIR = "./src/cut_detector/data/mid_bodies"

def main():
    argv = sys.argv
    argc = len(argv)

    if argc == 1:
        print_help()
    elif argc >= 2:
        cmd = argv[1]
        if cmd == "visualize" or cmd == "v":
            flags = argv[2:]
            if len(flags) == 0:
                print("starting visualizer with no flags")
            else:
                print(f"starting visualizer with flags: {flags}")

            # We are not importing it at the top because it requires
            # special dependencies that are not used by any other
            # packages.
            from .visualizer import run_app
            run_app(flags)

        elif cmd == "lt-debug":
            print("starting wrapper lt debug run")
            from .lt_debug_wrapper import start_wrapped_lt_run
            start_wrapped_lt_run()

        elif cmd == "w":
            from .w import run_app
            run_app()

        elif cmd == "gen-d-gt":
            print("generating detection ground truth")
            from .detection_gt import run_generate_ground_truth
            run_generate_ground_truth()

        elif cmd == "bench-d-gt":
            print("benching detection against ground truth")
            from .detection_gt import run_bench_detection
            run_bench_detection()

        else:
            raise RuntimeError(f"Unknown command {cmd}")


def print_help():
    print("===  'Next' workspace help  ===")
    print(dedent("""
        'bpkg' Blob Package is a collection of libraries and executables designed to improve 
        the performance of midbody detection and tracking."""))
    print("")
    print("To run a tool, you can use the syntax:")
    print("python next/main [CMD] <options...>")
    print("")
    print("available CMDs:")
    print("lt-debug: laptrack debug")
    print("gen-d-gt: generate detection ground truth (using defaults)")
    print("bench-d-gt: bench detection against ground truth (with defaults)")
    print("")
    print("The following are still WIP:")
    print("visualize | v: The detection visualizer")
    print("w: V2 of the visualizer")


if __name__ == "__main__":
    main()

