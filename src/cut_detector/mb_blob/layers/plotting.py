""" Plotting Layers
"""

import sys
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
        src_img = env.get(self.src_key)
        if src_img is None:
            print(f"Could not find key {self.src_key} in env:\n", env, file=sys.stderr)

        if self.sbs:
            raise RuntimeError("Side-By-Side (sbs) plotting not supported yet")
        else:
            fig, ax = plt.subplots()
            ax.imshow(src_img)
            for blob in blobs:
                y, x, r = blob
                circle = plt.Circle((x, y), r, linewidth=2, fill=False)
                ax.add_patch(circle)
        plt.show()