"""Playground to run cell tracking."""

import os
import pickle
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np

from cut_detector.data.tools import get_data_path
from cut_detector.factories.segmentation_tracking_factory import (
    SegmentationTrackingFactory,
)


def main(
    segmentation_results_path: Optional[str] = os.path.join(
        get_data_path("segmentation_results"), "example_video.bin"
    ),
):
    """
    Parameters
    ----------
    segmentation_results_path : str
        Path to the segmentation results, to avoid executing cellpose here.
    """
    # Load Cellpose results
    with open(segmentation_results_path, "rb") as f:
        cellpose_results = pickle.load(f)  # TYX

    # Perform tracking from Cellpose results
    factory = SegmentationTrackingFactory("")
    cell_spots, _ = factory.perform_tracking(
        cellpose_results,
        diam_labels=164,  # hardcoded since normally included in Cellpose model
    )

    # Display results
    for frame, cellpose_result in enumerate(cellpose_results):
        frame_cells = [cell for cell in cell_spots if cell.frame == frame]
        plt.imshow(cellpose_result)
        for local_cell in frame_cells:
            points = np.array(local_cell.spot_points)
            plt.plot(
                points[:, 0],
                points[:, 1],
                "o",
            )
            plt.plot(
                local_cell.x,
                local_cell.y,
                "x",
            )
        plt.show()


if __name__ == "__main__":
    main()
