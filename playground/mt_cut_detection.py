import os
import numpy as np
import pickle
from typing import Optional
from cnn_framework.utils.readers.tiff_reader import TiffReader
from matplotlib import pyplot as plt

from cut_detector.data.tools import get_data_path
from cut_detector.models.tools import get_model_path
from cut_detector.factories.mt_cut_detection_factory import (
    MtCutDetectionFactory,
)


def main(
    image_path: Optional[str] = get_data_path("videos"),
    mitosis_path: Optional[str] = get_data_path("mitoses"),
    scaler_path: Optional[str] = get_model_path("svc_scaler"),
    model_path: Optional[str] = get_model_path("svc_model"),
    hmm_bridges_parameters_file: Optional[str] = get_model_path(
        "hmm_bridges_parameters"
    ),
    display_svm_analysis=False,
    display_crops=True,
):
    # If paths are directories, take their first file
    if os.path.isdir(image_path):
        image_path = os.path.join(image_path, os.listdir(image_path)[0])
    if os.path.isdir(mitosis_path):
        mitosis_path = os.path.join(mitosis_path, os.listdir(mitosis_path)[0])

    # Read data: image, mitosis_track
    image = TiffReader(image_path, respect_initial_type=True).image  # TCZYX
    video = image.squeeze().transpose(0, 2, 3, 1)  # TYXC
    with open(mitosis_path, "rb") as f:
        mitosis_track = pickle.load(f)

    factory = MtCutDetectionFactory()

    results = factory.update_mt_cut_detection(
        mitosis_track,
        video,
        scaler_path,
        model_path,
        hmm_bridges_parameters_file,
    )

    if display_svm_analysis:
        # Plot 4 subplots
        _, axs = plt.subplots(2, 2)

        axs[0, 0].plot(np.array(results["distances"])[:, 0, 0])
        axs[0, 0].set_title("A vs rest")
        axs[0, 0].axhline(y=0, color="r", linestyle="-")

        axs[0, 1].plot(np.array(results["distances"])[:, 0, 1])
        axs[0, 1].set_title("B vs rest")
        axs[0, 1].axhline(y=0, color="r", linestyle="-")

        axs[1, 0].plot(np.array(results["distances"])[:, 0, 2])
        axs[1, 0].set_title("C vs rest")
        axs[1, 0].axhline(y=0, color="r", linestyle="-")

        axs[1, 1].plot(results["list_class_bridges"])
        axs[1, 1].plot(results["list_class_bridges_after_hmm"])
        axs[1, 1].set_title("Class bridges")

        plt.show()

    # Display series of crops
    for crop in results["crops"]:
        plt.figure()
        plt.imshow(crop, cmap="gray")
        plt.show()

if __name__ == "__main__":
    main()
