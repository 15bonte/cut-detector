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
from cut_detector.utils.bridges_classification.template_type import (
    TemplateType,
)


def re_organize_channels(image):
    """
    From any order to TYXC.
    """
    image = image.squeeze()
    assert image.ndim == 4, "Image must be 4D: time, channels, height, width"

    # Get dimension with minimum size and put last
    min_dim = np.argmin(image.shape)
    image = np.moveaxis(image, min_dim, -1)

    # Get dimension with minimum size in first 3 dimensions and put first
    min_dim = np.argmin(image.shape[:3])
    image = np.moveaxis(image, min_dim, 0)

    return image


def main(
    image_path: Optional[str] = get_data_path("videos"),
    mitosis_path: Optional[str] = get_data_path("mitoses"),
    scaler_path: Optional[str] = get_model_path("svc_scaler"),
    model_path: Optional[str] = get_model_path("svc_model"),
    hmm_bridges_parameters_file: Optional[str] = get_model_path(
        "hmm_bridges_parameters"
    ),
    display_svm_analysis=False,
    display_crops=False,
):
    # If paths are directories, take their first file
    if os.path.isdir(image_path):
        image_path = os.path.join(image_path, os.listdir(image_path)[0])
    if os.path.isdir(mitosis_path):
        mitosis_path = os.path.join(mitosis_path, os.listdir(mitosis_path)[0])

    # Read data: image, mitosis_track
    image = TiffReader(image_path, respect_initial_type=True).image
    video = re_organize_channels(image)  # TYXC
    with open(mitosis_path, "rb") as f:
        mitosis_track = pickle.load(f)

    factory = MtCutDetectionFactory(
        template_type=TemplateType.ALL,
        # coeff_height_peak=0.0
    )

    results = factory.update_mt_cut_detection(
        mitosis_track,
        video,
        scaler_path,
        model_path,
        hmm_bridges_parameters_file,
        debug_plot=True,
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

    if display_crops:
        # Display series of crops
        for crop in results["crops"]:
            plt.figure()
            plt.imshow(crop, cmap="gray")
            plt.show()


if __name__ == "__main__":
    IMAGE_PATH = r"C:\Users\thoma\data\Data Nathalie\videos_debug\20231019-t1_siSpastin-50-2.tif"
    MITOSIS_PATH = r"C:\Users\thoma\OneDrive\Bureau\mitoses\20231019-t1_siSpastin-50-2_mitosis_32_5_to_37.bin"

    main(IMAGE_PATH, MITOSIS_PATH)
    # main()
