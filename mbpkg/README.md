# MidBody PacKaGe (MBPKG)

## How it is meant to be used
For now, it is meant to be used as a collection of binaries that are compatible
with each other through the use of the same libraries.
To use mbpkg as a library itself, it will have to be moved in the src
directory (so that 'mbpkg' can be found in `sys.path` last entry, cut-detector/src)

## Content
MBPKG defines the following modules:
- env: environment constants that simplify writing binaries
- logger: like print, but does normal printing + logging to a file
- helpers: defines several helping modules (meant to be imported directly):
    - logging: a logger that prints and writes to a log file at the same time
    - json_processing: functions to extract and typecheck data from json
- detection_truth: detection ground truth: 
    - loading GT from JSON
    - object representation of ground truth
    - writing GT to JSON
- detectors:
    - loading detectors from JSON
    - representing detectors as objects with settings (instead of just functions)
    - generating detector functions at runtime from objects
    - writing detectors to JSON
- detection_stat: representation detection bench stat against ground truth:
    - object representation
    - benching a detector against GT
    - loading JSON data
    - writing JSON data
- detection_bench:
    - comparing detectors
    - printing/logging report

the bin package contains the following modules:
- gt_editor: detection annotation software
- detection_debugger: seeing the detection sigma map / normalization
- detection_bench: bench several pipelines, obtain statistics, generate graphs
- pipeline_run: runs a pipeline with debugging / logging (->pg/mid_body_detection)
- pipeline_bench: runs pipelines against the real ground truth (->pg/mid_body_{exec/eval})

These binaries are launched as CLI arguments to the module:
'python mbpkg {command}'
The syntax can be found by running mbpkg without arguments:
'python mbpkg'

## Changes from BPKG:
Libraries and binaries are now separated from one another:
Libraries are top level.

Binaries have been moved into a "binary" folder (like sandbox before).
Sandbow has been removed.
Binaries are meant to be modified. They usually have a list of constants
at the top.

Config files are probably removed (writing this before actually doing it).
These constants are now at the top of the binary files, or they can be shared
across several binaries

Along the libraries, there is an "env" module that contains constants
that can be used in the binaries (ie: source file paths, log/data dir paths).
Env can be modified too to add new values that can be shared across the
libaries.

Custom file (typically JSON files) now have a dedicated explanation file
that specifies the format rules (ie fields, types, values...)