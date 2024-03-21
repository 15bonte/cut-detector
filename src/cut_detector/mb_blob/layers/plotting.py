""" Plotting Layers
"""

import sys
import matplotlib.pyplot as plt
from .layer import BlobLayer

class PlotSaver(BlobLayer):
    def __init__(self, filedir: str, filename: str, add_src_fp: bool = False):
        self.filedir = filedir
        self.filename = filename
        self.add_src_fp = add_src_fp
    
    def apply(self, env: dict):
        src_fp = env["source_fp"].split("/")[-1]
        frame  = env["frame"]
        pipeline_name = env["pipeline_name"]
        if pipeline_name is None:
            pipeline_name = ""
        if src_fp is None:
            filepath = f"{self.filedir}/{self.filename}_{pipeline_name}_f{frame}.png"
        else:
            filepath = f"{self.filedir}/{src_fp}_{self.filename}_{pipeline_name}_f{frame}.png"
        plt.savefig(filepath)

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

    def __init__(
            self, 
            blobs_key: str = "blobs", 
            srcimg_key: str = "src", 
            sbs: bool = False, 
            title_prefix: str = "Blobs Found",
            plt_saver: PlotSaver | None = None,
            should_show: bool = True
            ):
        self.blobs_key = blobs_key
        self.src_key = srcimg_key
        self.sbs = sbs
        self.title_prefix = title_prefix
        self.plt_saver = plt_saver
        self.should_show = should_show
    
    def apply(self, env: dict):
        blobs = env.get(self.blobs_key, [])
        src_img = env.get(self.src_key)
        frame = env["frame"]
        if src_img is None:
            # print(f"Could not find key {self.src_key} in env:\n", env, file=sys.stderr)
            raise RuntimeError(f"Could not find key {self.src_key} in environment")

        if self.sbs:
            raise RuntimeError("Side-By-Side (sbs) plotting not supported yet")
        else:
            fig, ax = plt.subplots()
            ax.imshow(src_img)
            for blob in blobs:
                y, x, r = blob
                circle = plt.Circle((x, y), r, linewidth=2, fill=False)
                ax.add_patch(circle)

        plt.title(f"{self.title_prefix} frame {frame}")
        if not self.plt_saver is None:
            self.plt_saver.apply(env)
        
        if self.should_show:
            plt.show()
        else:
            plt.clf()