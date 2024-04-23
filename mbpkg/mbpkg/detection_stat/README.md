# Detection Statistics

## Content

Compute detection statistics based on accuracy against ground truth.

This module mainly defines the following:
- The function `generate_stats` that takes a detectors and a ground truth file.
  It computes time/accuracy statistics and return an instance of `DetectionStat`
- The class `DetectionStat`: a representation of the result of time and accuracy
  computations. This class class provides two methods:
    - `log_report`: prints to console and logs a report to a file
    - `write_stat`: write the stat to JSON format
- A function `load_detection_stat`: loads a detection stat from a json file


## File Format

The JSON file format used is the following:
- detector:
    - str
    - the str representation of a detector, that can be used to build another instance
      (see the mbpkg/detector package)
- source_path:
    - str
    - path to the source file, from the current working directory
- time:
    - float, optional
    - Time taken to run the detection
- fn_count:
    - int
    - The number of false negatives
- fp_count:
    - int
    - The number of false positives
- distances:
    - np.ndarray (list) int
    - A list of distances between between the detected points and the ground truth