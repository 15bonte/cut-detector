""" Pipeline execution.
Based function wants the table directly,
while exts functions takes a function path.
Maybe an ext function that triggers the rest of the pipeline (tracking could be added) ?
"""

import numpy as np
from .layers import BlobLayer

def run_pipeline(pipeline: list[BlobLayer], out: str, img: np.array, logging: bool = False):
    pass

