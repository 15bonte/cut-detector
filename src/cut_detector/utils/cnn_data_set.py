import numpy as np
import albumentations as A

from cnn_framework.utils.data_sets.abstract_data_set import AbstractDataSet
from cnn_framework.utils.data_sets.dataset_output import DatasetOutput
from cnn_framework.utils.tools import handle_image_type


class CnnDataSet(AbstractDataSet):
    """Custom class to avoid loading images from folder.

    Parameters
    ----------
    data : list[np.array]
        List of images to load.
    """

    def __init__(self, data: list[np.array], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = data

    def set_transforms(self) -> None:
        """Set the transforms to apply to the images."""
        height, width = self.params.input_dimensions.to_tuple(False)
        self.transforms = A.Compose(
            [
                A.Normalize(
                    self.mean_std["mean"],
                    std=self.mean_std["std"],
                    max_pixel_value=1,
                ),
                A.PadIfNeeded(
                    min_height=height,
                    min_width=width,
                    border_mode=0,
                    value=0,
                    p=1,
                ),
                A.CenterCrop(height=height, width=width, p=1),
            ]
        )

    def generate_images(self, filename: str) -> DatasetOutput:
        """Generate the images for the dataset.

        Parameters
        ----------
        filename : str
            Name of the file to load.

        Returns
        -------
        DatasetOutput
            Output of the dataset.
        """
        idx = int(filename.split(".")[0])
        # Get image and adapt it to torch
        nucleus_image = np.moveaxis(self.data[idx], 0, -1)  # YXC
        nucleus_image = handle_image_type(nucleus_image)  # to [0, 1]
        # Define any target value
        target_array = np.zeros(self.params.nb_classes)
        target_array[0] = 1
        # Construct output
        return DatasetOutput(
            input=nucleus_image,
            target_array=target_array,
        )
