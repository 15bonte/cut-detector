from matplotlib import pyplot as plt
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
    """
    Model manager for Binary Bridges CNN classification models.
    """

    def augment_image(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Function to augment image following binary bridges principle.
        """
        input_array = input_tensor.detach().cpu().numpy()  # BCYX

        augmentation_tool = MicroTubulesAugmentation()

        augmented_array = []
        for augmented_tensor in input_array:
            # Put first dimension last
            augmented_images = augmentation_tool.generate_augmentations(
                augmented_tensor
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
        """
        Function to generate outputs from inputs for given model.
        """
        augmented_input = self.augment_image(dl_element.input)

        dl_element.to_device(self.device)
        augmented_input = augmented_input.to(self.device)

        predictions = torch.softmax(
            self.model(augmented_input.float()), dim=-1
        )

        # From MT detection to cut detection
        predictions = predictions.view(-1, 4, 2, 2)  # 4*2 images, 2 scores
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

        # target_argmax = torch.argmax(dl_element.target, dim=1)
        # predictions_argmax = torch.argmax(final_predictions, dim=1)
        # predictions_to_print = predictions.view(-1, 8, 2)
        # for img_idx, original_image in enumerate(dl_element.input):
        #     if predictions_argmax[img_idx] == target_argmax[img_idx]:
        #         continue
        #     plt.subplot(1, 9, 1)
        #     mat_original = original_image[0].squeeze().detach().cpu().numpy()
        #     plt.title(
        #         f"{self.file_name_encoder.decode(dl_element.encoded_file_name[img_idx])} \n Pred {2 - predictions_argmax[img_idx].detach().cpu().numpy()} MT vs Target {2 - target_argmax[img_idx].detach().cpu().numpy()} MT"
        #     )
        #     plt.imshow(mat_original, cmap="gray")
        #     for plt_idx, idx in enumerate(
        #         range(8 * img_idx, 8 * (img_idx + 1))
        #     ):
        #         plt.subplot(1, 9, 2 + plt_idx)
        #         mat_sub_image = (
        #             augmented_input[idx][0].squeeze().detach().cpu().numpy()
        #         )
        #         augment_pred = (
        #             predictions_to_print[img_idx][plt_idx]
        #             .detach()
        #             .cpu()
        #             .numpy()
        #         )
        #         predicted_category = ["No MT", "MT"][np.argmax(augment_pred)]
        #         plt.title(f"{predicted_category} {str(max(augment_pred))[:4]}")
        #         plt.imshow(mat_sub_image, cmap="gray")

        #     # Display on whole screen
        #     figManager = plt.get_current_fig_manager()
        #     figManager.window.showMaximized()
        #     plt.show()

        # Update metric
        dl_metric.update(
            final_predictions,
            dl_element.target,
            adds=dl_element.additional,
            mean_std=data_loader.dataset.mean_std,
        )
