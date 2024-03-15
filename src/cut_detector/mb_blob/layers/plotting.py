""" Plotting Layers
"""

import matplotlib.pyplot as plt
from .layer import BlobLayer

class PlotImage(BlobLayer):
    """Plots a simple image, by default 'img' but that can be changed
    
    For now, image is shown in a blocking way
    """
    def __init__(self, label: str = "Figure", image_key: str = "img"):
        self.image_key = image_key
        self.image_label = label
    
    def apply(self, env: dict):
        frame_idx = env["frame"]
        plt.imshow(env[self.image_key])
        plt.title(f"{self.image_label}: {frame_idx}")
        plt.show()


class PlotBlobs(BlobLayer):
    """Plots the blobs followinf LapOfGauss shape

    Plotting is done in a blocking way
    """

    def __init__(self, blobs_key: str = "blobs", srcimg_key: str = "src", sbs: bool = False):
        self.blobs_key = blobs_key
        self.src_key = srcimg_key
        self.sbs = sbs
    
    def apply(self, env: dict):
        blobs = env.get(self.blobs_key, [])
        fig, ax = plt.subplots()
        ax.imshow(self.src_key)
        for blob in blobs:
            y, x, r = blob
            circle = plt.Circle((x, y), r, linewidth=2, fill=False)
            ax.add_patch(circle)
        plt.show()