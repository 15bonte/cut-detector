"""Playground to run cell segmentation."""

import os
from typing import Optional
import matplotlib.pyplot as plt

from cnn_framework.utils.readers.tiff_reader import TiffReader

from cut_detector.data.tools import get_data_path
from cut_detector.models.tools import get_model_path
from cut_detector.factories.segmentation_tracking_factory import (
    SegmentationTrackingFactory,
)


def main(
    image_path: Optional[str] = get_data_path("videos"),
    model_path: Optional[str] = os.path.join(
        get_model_path("segmentation"), "segmentation_model"
    ),
):
    """
    Parameters
    ----------
    image_path : str
        Path to the image to process.
    model_path : str
        Path to the segmentation model.
    """
    # If image_path or model_path are directories, take their first file
    if os.path.isdir(image_path):
        image_path = os.path.join(image_path, os.listdir(image_path)[0])
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, os.listdir(model_path)[0])

    # Read image and preprocess if needed
    image = TiffReader(image_path, respect_initial_type=True).image  # TCZYX
    image = image.squeeze()  # TCYX

    # Initialize factory to get constants
    factory = SegmentationTrackingFactory(model_path)

    results, flows, _ = factory.perform_segmentation(image)  # TCYX

    for frame in range(image.shape[0]):
        # Plot image_to_segment and segmented_image
        _, ax = plt.subplots(2, 2)
        ax[0, 0].set_title("Raw image")
        ax[0, 0].imshow(image[frame, 2].squeeze(), cmap="gray")
        ax[0, 1].set_title("Cellpose")
        ax[0, 1].imshow(results[frame], cmap="viridis")
        ax[1, 0].set_title("Flow")
        ax[1, 0].imshow(flows[0][frame])
        ax[1, 1].set_title("Cell probability")
        ax[1, 1].imshow(flows[2][frame], cmap="gray")
        plt.show()


if __name__ == "__main__":
    main()
