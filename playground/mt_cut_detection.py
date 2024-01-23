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
from cut_detector.utils.mitosis_track import MitosisTrack


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
    display_intensity_analysis=True,
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
        mitosis_track: MitosisTrack = pickle.load(f)

    template_type = TemplateType.AVERAGE_CIRCLE
    factory = MtCutDetectionFactory(
        template_type=template_type,
    )

    results = factory.update_mt_cut_detection(
        mitosis_track,
        video,
        scaler_path,
        model_path,
        hmm_bridges_parameters_file,
        debug_plot=False,
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

    if display_intensity_analysis and len(results["templates"]) > 0:
        assert template_type == TemplateType.AVERAGE_CIRCLE

        # Plot 2 subplots
        _, axs = plt.subplots(2, 1)

        # Possible that first frames are skipped
        nb_points = len(np.array(results["templates"])[..., 0].squeeze())
        frame_indexes = list(mitosis_track.mid_body_spots)[-nb_points:]
        axs[0].plot(
            frame_indexes,
            np.array(results["templates"])[..., 0].squeeze(),
        )
        axs[0].set_title("First MT intensity")
        if mitosis_track.gt_key_events_frame is not None:
            axs[0].axvline(
                x=mitosis_track.gt_key_events_frame["first_mt_cut"],
                color="r",
                linestyle="-",
            )
            axs[0].axvline(
                x=mitosis_track.gt_key_events_frame["second_mt_cut"],
                color="r",
                linestyle="-",
            )

        axs[1].plot(
            frame_indexes,
            np.array(results["templates"])[..., 3].squeeze(),
        )
        axs[1].set_title("Second MT intensity")
        if mitosis_track.gt_key_events_frame is not None:
            axs[1].axvline(
                x=mitosis_track.gt_key_events_frame["first_mt_cut"],
                color="r",
                linestyle="-",
            )
            axs[1].axvline(
                x=mitosis_track.gt_key_events_frame["second_mt_cut"],
                color="r",
                linestyle="-",
            )

        plt.show()


if __name__ == "__main__":
    # IMAGE_PATH = r"C:\Users\thoma\data\Data Nathalie\videos_debug\20231019-t1_siSpastin-50-2.tif"
    # MITOSIS_PATH = r"C:\Users\thoma\OneDrive\Bureau\mitoses\20231019-t1_siSpastin-50-2_mitosis_32_5_to_37.bin"
    # main(IMAGE_PATH, MITOSIS_PATH)

    FOLDER_MITOSIS = r"C:\Users\thoma\data\Data Nathalie\mitoses"
    FOLDER_VIDEO = r"C:\Users\thoma\data\Data Nathalie\videos"
    for mitosis_file in os.listdir(FOLDER_MITOSIS):
        mitosis_path = os.path.join(FOLDER_MITOSIS, mitosis_file)
        image_name = mitosis_file.split("_mitosis")[0]
        image_path = os.path.join(FOLDER_VIDEO, image_name + ".tif")
        main(image_path, mitosis_path)

    # main()
