# Detection Truth


## What this defines
Defines a representation for the detection ground truth, called `DetectionGT`.

This representation can be used in 3 ways:
- loaded from a file: see `load_gt`
- Created at runtime, see `DetectionGT.__init__`
- written to a file: see `DetectionGT.write`

(choice to use a function to open VS a method to write has been made to follow
Python's `open` function VS `write`/`close` file method).

Additionnaly, the format used for serialization is described in this document,
see below.


## What is not defined
This package does not define a way to measure accuracy to ground truth.


## Serialization format (version 0.2.0)
Ground Truth is serialized to JSON.
The keys are the following:

- file:
    - str
    - value: path to the source file in the current working directory.
- detection_method:
    - str
    - value: the name of the method to generate this source file. Can be
      anything. No defined role, can be used to provide a better user
      output
- spots:
    - dict[frame_idx: str, list[dict{x:int, y:int}]]
    - value: a dictionnary of frame_idx / list of points. 
      frame_idx is an index so frame 1 is index 0.
      If a frame has no point, the frame_idx/list pair can be
      either omitted or indicated by a key associated with an empty list

Extra fields, if any, are ignored