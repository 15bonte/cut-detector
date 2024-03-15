""" The layesr responsible for saving. Saving here does not mean
writing to disk.
It juste means calling the save handler(s)
"""

import sys
from time import time
from .layer import BlobLayer
from ..savers import Saver


class SaveStdBlobMetrics(BlobLayer):
    """Saves the standard metrics for the blobs processes.
    
    More specifically, it saves:
    - source file, if any
    - Difference of current time and the 'time' environment field
    - frame number
    - blobs yxs
    """
    def __init__(self):
        pass

    def apply(self, env: dict):
        ## time
        cur_time = time()
        reference_time = env["time"]
        delta = cur_time - reference_time

        ## source path
        source_fp = env.get("source_fp")

        ## frame
        frame = env["frame"]

        ## blobs
        blobs = env.get("blobs", []) # YXS

        ## saving
        savers: list[Saver] = env.get("savers", [])
        for s in savers:
            for b in blobs:
                s.save({
                    "source_fp": source_fp,
                    "frame": frame, 
                    "time": delta,
                    "x": b[1],
                    "y": b[0],
                    "s": b[2]
                })
