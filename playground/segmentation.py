import os
from typing import Optional
import torch
import matplotlib.pyplot as plt
from cellpose import models

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
    diameter=0,  # 0 means using segmentation model saved value
    channel_to_segment=3,  # index starts at 1
    nucleus_channel=0,  # 0 means no nucleus channel
):
    """
    Script to run simple cellpose segmentation.
    """
    # If image_path or model_path are directories, take their first file
    if os.path.isdir(image_path):
        image_path = os.path.join(image_path, os.listdir(image_path)[0])
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, os.listdir(model_path)[0])

    # Read image and preprocess if needed
    image = TiffReader(image_path).image  # TCZYX

    # Initialize factory to get constants
    factory = SegmentationTrackingFactory(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.CellposeModel(pretrained_model=[model_path], device=device)

    for frame in range(image.shape[0]):
        frame_image = image[frame, ...].squeeze()
        results, flows, _ = model.eval(
            [frame_image],
            channels=[channel_to_segment, nucleus_channel],
            diameter=diameter,
            flow_threshold=factory.flow_threshold,
            cellprob_threshold=factory.cellprob_threshold,
            augment=factory.augment,
        )

        # Plot image_to_segment and segmented_image
        _, ax = plt.subplots(2, 2)
        ax[0, 0].set_title("Raw image")
        ax[0, 0].imshow(frame_image[channel_to_segment - 1], cmap="gray")
        ax[0, 1].set_title("Cellpose")
        ax[0, 1].imshow(results[0], cmap="viridis")
        ax[1, 0].set_title("Flow")
        ax[1, 0].imshow(flows[0][0])
        ax[1, 1].set_title("Cell probability")
        ax[1, 1].imshow(flows[0][2], cmap="gray")
        plt.show()


if __name__ == "__main__":
    main()
