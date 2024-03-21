""" Pipeline execution.
Based function wants the table directly,
while exts functions takes a function path.
Maybe an ext function that triggers the rest of the pipeline (tracking could be added) ?
"""

import numpy as np
from .layers import BlobLayer
from .savers import Saver

def run_pipeline(
        pipeline: list[BlobLayer], 
        out: str, 
        movie: np.array, 
        src_fp: str | None = None,
        logging: bool = False,
        savers: list[Saver] = [],
        pipeline_name: str | None = None
        ):
    """Executes the pipeline on the movie data.
    logging controls whether some simple info messages are sent or not.
    out allows you to choose which env field is sent back as a return value.
    Currently, only a single field can be sent.

    movie is expected to have shape [frame, ...] where "..." are data values (like pixels)
    """
    # Initial environment setup
    env = {
        "frame": 0,
        "img": None,
        "savers": savers,
        "source_fp": src_fp,
        "pipeline_name": pipeline_name
    }

    # setting frame number and looping over frames
    frame_count = movie.shape[0]
    for frame_idx in range(frame_count):
        if logging: print(f"video {src_fp}, frame {frame_idx}:")
        env["frame"] = frame_idx
        env["img"] = movie[frame_idx, :, :]
        for layer in pipeline:
            layer.apply(env=env)

    # savers final save call
    for s in savers:
        s.delayed_save()

    # Return value
    out_v = env.get(out)
    if out_v is None:
        raise RuntimeError(f"key {out} not found in env, among:\n{env}")
    if logging: print(f"Done, returning {out}")
    return out_v

