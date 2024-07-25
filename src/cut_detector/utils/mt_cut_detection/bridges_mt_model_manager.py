import numpy as np
import torch
from torch.utils.data import DataLoader

from cnn_framework.utils.metrics.abstract_metric import AbstractMetric
from cnn_framework.utils.data_sets.dataset_output import DatasetOutput
from cnn_framework.utils.model_managers.cnn_model_manager import (
    CnnModelManager,
)

from .micro_tubules_augmentation import (
    MicroTubulesAugmentation,
)


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

        augmented_array = []
        for input_array in input_arrays:
            # Put first dimension last
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
