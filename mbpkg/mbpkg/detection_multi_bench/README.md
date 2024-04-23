# Detection Multi Bench Package


## Content

Bench several detectors against one another, against the same ground truth.

Main elements are:
- `generate_multi_stat_bench`: runs the detectors against the same ground truth
- `MultiBenchStat`: the representation of these runs:
    - a method: `write_result`: to write the JSON result to a file
- `load_multi_bench_stat`: loads the reprentation from a JSON file. 


## File format
MultiBenchStat is stored in JSON with the following entries:
- source_path:
    - str
    - The path to the mitosis source file in the current working directory
- distances:
    - dict[str, list[float]]
    - the distances associated with their detector.
- fn_counts:
    - dict[str, int]
    - the number of false negatives (misses points) associated with their detector (fmtstr)
- fp_counts:
    - dict[str, int]
    - the number of false positives (extra points) associated with their detector (fmtstr)
- times:
    - dict[str, float]
    - the execution time (in seconds) associated with their detector.
