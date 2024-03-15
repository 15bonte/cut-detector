""" Layers that controls the environment, or whose main purpose
is to act based on the environment
"""

from time import time
from .layer import BlobLayer


class OnFrame(BlobLayer):
    """Controls the activation of a layer based on the current frame:
    If the current frame DOES match the condition, the layer is activated.
    If it does not and the off_layer is defined, then the off_layer is called.
    """
    def __init__(self, frames: list[int], on_layer: BlobLayer | None, off_layer: BlobLayer | None = None):
        self.frames = frames
        self.on_layer = on_layer
        self.off_layer = off_layer
    
    def apply(self, env: dict):
        current_frame = env["frame"]
        if current_frame in self.frames:
            if self.on_layer is not None:
                self.on_layer.apply(env)
        elif self.off_layer is not None:
            self.off_layer.apply(env)


class WriteImg(BlobLayer):
    """Copies the current img environment field in another variable"""
    def __init__(self, as_key: str):
        self.as_key = as_key
    
    def apply(self, env: dict):
        env[self.as_key] = env["img"]


class WriteTime(BlobLayer):
    """Writes the current time in the time field of the environment"""
    def __init__(self, as_key: str = "time"):
        self.as_key = as_key

    def apply(self, env: dict):
        env[self.as_key] = time()