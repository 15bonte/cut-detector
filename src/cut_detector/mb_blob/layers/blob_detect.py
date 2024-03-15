""" The Gaussian Layers
"""

import matplotlib.pyplot as plt
from skimage.feature import blob_log
from .layer import BlobLayer
from .lapgau_helpers import blob_log_with_plotting, BlobLogVisuSettings

class LapOfGauss(BlobLayer):
    """Applies a Laplacian of Gaussian to detect blobs"""
    def __init__(self, min_sig: int, max_sig: int, n_sig: int, threshold: float = 0.2, visu: BlobLogVisuSettings = None) -> None:
        self.min_sig = min_sig
        self.max_sig = max_sig
        self.n_sig = n_sig
        self.threshold = threshold
        if visu is None:
            self.visu = BlobLogVisuSettings()
        else:
            self.visu = visu

    def apply(self, env: dict):
        img = env["img"]
        if self.visu is None:
            blobs = blob_log(
                img, 
                min_sigma=self.min_sig,
                max_sigma=self.max_sig,
                num_sigma=self.n_sig,
                threshold=self.threshold
            )
        else:
            blobs = blob_log_with_plotting(
                img, 
                min_sigma=self.min_sig,
                max_sigma=self.max_sig,
                num_sigma=self.n_sig,
                threshold=self.threshold,
                plot_settings=self.visu,
            )
            if self.visu.cube_plotting_threshold is not None or len(self.visu.sig_layers_idx) != 0:
                plt.show()
                
        env["blobs"] = blobs

class DiffOfGauss(BlobLayer):
    """Applies a Difference of Gaussian to detect blobs"""
    pass

class DetOfHess(BlobLayer):
    """Applies a Difference of Gaussian to detect blobs"""
    pass