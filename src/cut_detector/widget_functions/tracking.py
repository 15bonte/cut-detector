import os
from typing import Optional

from ..factories.segmentation_tracking_factory import (
    SegmentationTrackingFactory,
)
from ..models.tools import get_model_path


def perform_tracking(
    video_path: str,
    fiji_path: str,
    save_folder: str,
    model_path: Optional[str],
    fast_mode: Optional[bool] = False,
) -> None:
    # Create save directory if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if model_path is None:
        model_path = os.path.join(
            get_model_path("segmentation"), "segmentation_model"
        )

    segmentation_tracking_factory = SegmentationTrackingFactory(model_path)
    segmentation_tracking_factory.perform_trackmate_tracking(
        video_path,
        fiji_path,
        save_folder,
        fast_mode,
    )
