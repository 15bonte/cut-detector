""" Mini TIFF Reader.
Based on cnn_framework.utils.reader.TiffReader.

Idea here is to make a mini function for use cases that don't need
the additional features provided by AbstractReader.
"""

from typing import Optional

import numpy as np
from aicsimageio import AICSImage

def mini_read_tiff(filepath: str) -> np.ndarray:
    aics_img = AICSImage(filepath)
    image = reorganize_channels(
        aics_img.data, "TCZYX", aics_img.dims.order
    )
    return image

def reorganize_channels(
        image: np.ndarray,
        target_order: Optional[str],
        original_order: Optional[str],
    ):
        """
        Make sure image dimensions order matches dim_order.
        """
        # Add missing dimensions if necessary
        for dim in target_order:
            if dim not in original_order:
                original_order = dim + original_order
                image = np.expand_dims(image, axis=0)

        indexes = [original_order.index(dim) for dim in target_order]
        return np.moveaxis(image, indexes, list(range(len(target_order))))


