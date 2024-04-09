""" Layers that controls the environment
"""

from time import time
from .layer import BlobLayer

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