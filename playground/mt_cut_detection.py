"""Playground to run micro-tubules cut detection on a single mitosis.
Predicted classes and bridge crops are displayed."""

import os
from typing import Optional
from matplotlib import pyplot as plt

from cnn_framework.utils.readers.tiff_reader import TiffReader

from cut_detector.data.tools import get_data_path
from cut_detector.models.tools import get_model_path
from cut_detector.factories.mt_cut_detection_factory import (
    MtCutDetectionFactory,
)
from cut_detector.utils.mitosis_track import MitosisTrack
from cut_detector.utils.tools import re_organize_channels


def main(
    image_path: Optional[str] = get_data_path("videos"),
    mitosis_path: Optional[str] = get_data_path("mitoses"),
    hmm_bridges_parameters_file: Optional[str] = os.path.join(
        get_model_path("hmm"), "hmm_bridges_parameters.npz"
    ),
    bridges_mt_cnn_model_path: Optional[str] = get_model_path(
        "bridges_mt_cnn"
    ),
    display_predictions_analysis=True,
    display_crops=True,
):
    """
    Parameters
    ----------
    image_path : str
        Path to the image to process.
    mitosis_path : str
        Path to the mitoses data.
    hmm_bridges_parameters_file : str
        Path to the HMM parameters file.
    bridges_mt_cnn_model_path : str
        Path to the bridges MT CNN model.
    display_predictions_analysis : bool
        If True, display the predicted classes before and after the HMM.
    display_crops : bool
        If True, display the crops of the bridges.
    """
    # If paths are directories, take their first file
    if os.path.isdir(image_path):
        image_path = os.path.join(image_path, os.listdir(image_path)[0])
    if os.path.isdir(mitosis_path):
        mitosis_path = os.path.join(mitosis_path, os.listdir(mitosis_path)[0])

    # Read data: image, mitosis_track
    image = TiffReader(image_path, respect_initial_type=True).image
    with open(mitosis_path, "rb") as f:
        mitosis_track = MitosisTrack.load(f)

    image = re_organize_channels(image.squeeze())  # TYXC

    factory = MtCutDetectionFactory()

    results = factory.update_mt_cut_detection(
        [mitosis_track],
        image,
        hmm_bridges_parameters_file,
        bridges_mt_cnn_model_path,
        debug_mode=True,
    )

    if display_predictions_analysis:
        plt.plot(results["predictions"][mitosis_track.id])
        if (
            mitosis_track.id in results["predictions_after_hmm"]
        ):  # if classification possible
            plt.plot(results["predictions_after_hmm"][mitosis_track.id])
        plt.title("Class bridges")
        plt.show()

    if display_crops:
        # Display series of crops
        for crop in results["images"][mitosis_track.id]:
            plt.figure()
            plt.imshow(crop[0], cmap="gray")
            plt.show()


if __name__ == "__main__":
    main()
