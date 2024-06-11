from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import cv2
from scipy import ndimage

from cnn_framework.utils.metrics.abstract_metric import AbstractMetric
from cnn_framework.utils.data_sets.dataset_output import DatasetOutput
from cnn_framework.utils.model_managers.cnn_model_manager import (
    CnnModelManager,
)

from .micro_tubules_augmentation import (
    MicroTubulesAugmentation,
)


def transform_image_orientation(
    image, margin=10, percentile=75
) -> tuple[np.ndarray]:
    """Compute image orientation and rotate it to have the bridges horizontal.

    Parameters
    ----------
    image : np.ndarray
        Image to rotate.
    margin : int, optional
        Margin to consider for orientation computation, by default 10.
    percentile : int, optional
        Percentile to consider for orientation computation, by default 75.

    Returns
    -------
    tuple[np.ndarray]
        Rotated image and binary image.
    """
    keep_3_channels = False
    if len(image.shape) == 3:
        assert image.shape[0] == 1, "Only one channel image is supported"
        image = image[0]
        keep_3_channels = True

    original_height, original_width = image.shape

    # Select sub-image (containing information)
    sub_image = image[
        original_height // 2 - margin : original_height // 2 + margin,
        original_width // 2 - margin : original_width // 2 + margin,
    ]
    threshold = np.percentile(sub_image, percentile)
    binary_image = (sub_image > threshold).astype(np.uint8)

    # Compute image moments
    moments = cv2.moments(binary_image)
    # Compute orientation
    orientation = 0.5 * np.arctan2(
        2 * moments["mu11"], moments["mu20"] - moments["mu02"]
    )
    # Compute orientation in degrees
    orientation_degrees = np.degrees(orientation)

    # Rotate image and keep original size
    rotated_image = ndimage.rotate(
        image, orientation_degrees, cval=image.min()
    )
    height, width = rotated_image.shape
    rotated_image = rotated_image[
        (height - original_height) // 2 : (height - original_height) // 2
        + original_height,
        (width - original_width) // 2 : (width - original_width) // 2
        + original_width,
    ]

    # Expand first dim
    if keep_3_channels:
        rotated_image = np.expand_dims(rotated_image, axis=0)

    return rotated_image, binary_image


class BridgesMtModelManager(CnnModelManager):
    """Model manager for Binary Bridges CNN classification models."""

    def augment_image(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Function to augment image following binary bridges principle.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor to augment.

        Returns
        -------
        torch.Tensor
            Augmented tensor.
        """
        input_arrays = input_tensor.detach().cpu().numpy()  # BCYX

        augmentation_tool = MicroTubulesAugmentation()
        augmentation_tool.augmentations.pop("top")
        augmentation_tool.augmentations.pop("bottom")

        augmented_array = []
        for orig_input_array in input_arrays:
            # Put first dimension last
            input_array, _ = transform_image_orientation(orig_input_array)
            augmented_images = augmentation_tool.generate_augmentations(
                input_array
            )
            for augmented_dict in augmented_images.values():
                # Put last dimension first
                augmented_image = augmented_dict["image"]  # YXC
                augmented_image = augmented_image.transpose(2, 0, 1)  # CYX
                augmented_array.append(augmented_image)

        augmented_array = np.array(augmented_array)
        augmented_tensor = torch.from_numpy(augmented_array)
        return augmented_tensor

    def model_prediction(
        self,
        dl_element: DatasetOutput,
        dl_metric: AbstractMetric,
        data_loader: DataLoader,
    ) -> None:
        """Function to generate outputs from inputs for given model.

        Parameters
        ----------
        dl_element : DatasetOutput
            Element from the data loader.
        dl_metric : AbstractMetric
            Metric to update.
        data_loader : DataLoader
            Data loader.
        """
        augmented_input = self.augment_image(dl_element.input)
        augmentation_nb = augmented_input.shape[0] // dl_element.input.shape[0]

        dl_element.to_device(self.device)
        augmented_input = augmented_input.to(self.device)

        predictions = torch.softmax(
            self.model(augmented_input.float()), dim=-1
        )

        # From MT detection to cut detection
        predictions = predictions.view(
            -1, augmentation_nb // 2, 2, 2
        )  # augmentation_nb images, 2 scores
        final_predictions = torch.zeros_like(dl_element.target)
        for idx, image_predictions in enumerate(predictions):
            for pair_predictions in image_predictions:
                # NB: no cut means 2 MT
                no_cut_prediction = (
                    pair_predictions[0][1] * pair_predictions[1][1]
                )
                cut_prediction = (
                    pair_predictions[0][1] * pair_predictions[1][0]
                    + pair_predictions[0][0] * pair_predictions[1][1]
                )
                two_cuts_prediction = (
                    pair_predictions[0][0] * pair_predictions[1][0]
                )
                final_predictions[idx][0] += no_cut_prediction
                final_predictions[idx][1] += cut_prediction
                final_predictions[idx][2] += two_cuts_prediction
        dl_element.prediction = final_predictions

        # Update metric
        dl_metric.update(
            final_predictions,
            dl_element.target,
            adds=dl_element.additional,
            mean_std=data_loader.dataset.mean_std,
        )
