""" Layers that allow one to control layer execution flow
"""

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


class Sequence(BlobLayer):
    """ Allows one to package several layers as one, to execute
    several layers based on a single condition
    """
    def __init__(self, *layers: BlobLayer):
        self.layers = layers
    
    def apply(self, env: dict):
        for layer in self.layers:
            layer.apply(env)