"""Run with binary_bridges_cnn_inference environment to generate semi images."""

import os

from cnn_framework.utils.tools import save_tiff
from cnn_framework.utils.readers.tiff_reader import TiffReader


from developers.ground_truth_generation.bridges.semi_images_generation.micro_tubules_augmentations_advanced import (
    MicroTubulesAugmentationAdvanced,
)
from developers.ground_truth_generation.bridges.semi_images_generation.mt_cut_detection_factory_advanced import (
    MtCutDetectionFactoryAdvanced,
)
from developers.training_and_evaluation.binary_bridges_cnn.bridges_parser import (
    BridgesParser,
)
from developers.training_and_evaluation.binary_bridges_cnn.model_params import (
    BinaryBridgesModelParams,
)


def main(params, save_dir, debug_plot, circle_radius=11, diff_radius=4.54):

    factory = MtCutDetectionFactoryAdvanced()
    file_paths = os.listdir(params.data_dir)

    # Create the save directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)

    for file_path in file_paths:
        # Read image
        file_path = os.path.join(params.data_dir, file_path)
        image = TiffReader(file_path).image.squeeze()  # CYX

        # Get the bridge type
        bridge_type = file_path.split("_")[-1].split(".")[0][-1]
        # string to int
        bridge_type = int(bridge_type)
        class_mode = bridge_type % 5

        # Create the coordinates list of the circle around the mid body spot
        # NB: first circle in list is the middle circle
        list_radius = [
            circle_radius,
            circle_radius + 0.25 * diff_radius,
            circle_radius - 0.25 * diff_radius,
            circle_radius + 0.5 * diff_radius,
            circle_radius - 0.5 * diff_radius,
            circle_radius + 0.75 * diff_radius,
            circle_radius - 0.75 * diff_radius,
            circle_radius + diff_radius,
            circle_radius - diff_radius,
        ]

        # Apply image pre-processing
        sir_tubulin_image = image[0, ...].squeeze()  # YX

        # Get useful data for each circle
        all_intensities = []
        for radius in list_radius:
            intensities = factory.get_circle_data(radius, sir_tubulin_image)
            all_intensities.append(intensities)

        # Compute 2 best peaks on average circle
        average_circle_peaks = factory.get_average_circle_peaks(
            all_intensities, debug_plot=debug_plot
        )

        # Save fake one cut image
        if save_dir is not None:
            file_name = os.path.basename(file_path).split("_c")[0]

            # Keep number of peaks depending on class mode
            if class_mode in (1, 3):
                average_circle_peaks = average_circle_peaks[:1]
            if class_mode in (2, 4):
                average_circle_peaks = []

            micro_tubules_augmentation = MicroTubulesAugmentationAdvanced(
                average_circle_peaks
            )
            augmentations = micro_tubules_augmentation.generate_augmentations(
                image
            )
            for title, title_values in augmentations.items():
                category = title_values["category"]
                save_path = os.path.join(
                    save_dir, f"{file_name}_{title}_c{category}.tiff"
                )
                save_tiff(
                    title_values["image"], save_path, original_order="YXC"
                )


if __name__ == "__main__":
    parser = BridgesParser()
    args = parser.arguments_parser.parse_args()

    parameters = BinaryBridgesModelParams()
    parameters.update(args)

    SAVE_DIR = parameters.data_dir + "_semi_images"
    DEBUG_PLOT = False

    main(parameters, save_dir=SAVE_DIR, debug_plot=DEBUG_PLOT)
