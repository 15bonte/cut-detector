import os
from typing import Optional
import pickle
import numpy as np
from matplotlib import pyplot as plt

from cnn_framework.utils.readers.tiff_reader import TiffReader

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
    hmm_bridges_parameters_file: Optional[str] = os.path.join(
        get_model_path("hmm"), "hmm_bridges_parameters.npz"
    ),
    bridges_mt_cnn_model_path: Optional[str] = get_model_path(
        "bridges_mt_cnn"
    ),
    display_predictions_analysis=True,
    display_crops=True,
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
        mitosis_track.adapt_deprecated_attributes()

    template_type = TemplateType.AVERAGE_CIRCLE
    factory = MtCutDetectionFactory(
        template_type=template_type,
    )

    results = factory.update_mt_cut_detection(
        [mitosis_track],
        video,
        hmm_bridges_parameters_file,
        bridges_mt_cnn_model_path,
        debug_mode=True,
    )

    if display_predictions_analysis:
        plt.plot(results["list_class_bridges"][mitosis_track.id])
        if (
            mitosis_track.id in results["list_class_bridges_after_hmm"]
        ):  # if classification possible
            plt.plot(results["list_class_bridges_after_hmm"][mitosis_track.id])
        plt.title("Class bridges")
        plt.show()

    if display_crops:
        # Display series of crops
        for crop in results["crops"][mitosis_track.id]:
            plt.figure()
            plt.imshow(crop[0], cmap="gray")
            plt.show()


if __name__ == "__main__":
    FOLDER_MITOSIS = r"C:\Users\thoma\data\Data Nathalie\mitoses"
    FOLDER_VIDEO = r"C:\Users\thoma\data\Data Nathalie\videos"
    for mitosis_file in os.listdir(FOLDER_MITOSIS):
        local_mitosis_path = os.path.join(FOLDER_MITOSIS, mitosis_file)
        image_name = mitosis_file.split("_mitosis")[0]
        local_image_path = os.path.join(FOLDER_VIDEO, image_name + ".tif")
        main(local_image_path, local_mitosis_path)

    # main()
