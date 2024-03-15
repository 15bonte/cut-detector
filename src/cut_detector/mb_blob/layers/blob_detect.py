""" The Gaussian Layers
"""

# from skimage.morphology import extrema, opening
from skimage.feature import blob_log
from .layer import BlobLayer

class LapOfGaussVisu:
    pass

class LapOfGauss(BlobLayer):
    """Applies a Laplacian of Gaussian to detect blobs"""
    def __init__(self, min_sig: int, max_sig: int, n_sig: int, threshold: float = 0.2, visu: LapOfGaussVisu = None) -> None:
        self.min_sig = min_sig
        self.max_sig = max_sig
        self.n_sig = n_sig
        self.threshold = threshold
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
            blobs = None
            raise RuntimeError("Visualisation for blob_log not available for now")
        env["blobs"] = blobs

class DiffOfGauss(BlobLayer):
    """Applies a Difference of Gaussian to detect blobs"""
    pass

class DetOfHess(BlobLayer):
    """Applies a Difference of Gaussian to detect blobs"""
    pass