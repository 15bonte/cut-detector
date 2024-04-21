LIST_MESSAGE = """
--- Binary list --
(to run one, type 'python mbpkg <name>')

d-debug:   opens the detection debugger tool
gt-editor: open the ground truth editor tool
d-bench:   run the detection benchmark
p-bench:   runs the pipeline benchmark
p-run:     simple pipeline run
"""
def run_binaries(name: str):
    if name in ["list", "l", "--list", "-l"]:
        print(LIST_MESSAGE)

    elif name == "d-debug":
        from bin.detection_debugger import run_app
        run_app()

    elif name == "gt-editor":
        from bin.gt_editor import run_app
        run_app()

    elif name == "d-bench":
        from bin.detection_bench import run_detection_bench
        run_detection_bench()

    elif name == "p-bench":
        from bin.pipeline_bench import run_pipeline_bench
        run_pipeline_bench()

    elif name == "p-run":
        from bin.pipeline_run import run_pipeline_run
        run_pipeline_run()

    elif name == "vis":
        from bin.visualizer import drive_app
        drive_app()
        
    else:
        print(f"Unknown binary {name}, see list 'python mbpkg l'")