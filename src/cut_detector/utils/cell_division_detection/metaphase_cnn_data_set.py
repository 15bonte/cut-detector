import numpy as np
import albumentations as A

from cnn_framework.utils.data_sets.abstract_data_set import AbstractDataSet
from cnn_framework.utils.data_sets.dataset_output import DatasetOutput
from cnn_framework.utils.preprocessing import normalize_array


class MetaphaseCnnDataSet(AbstractDataSet):
    """
    Custom class to avoid loading images from folder.
    """

    def __init__(self, data, *args, **kwargs):
        # Not pythonic, but needed as super init calls generate_raw_images
        self.data = data
        super().__init__(*args, **kwargs)

    def set_transforms(self):
        height, width = self.params.input_dimensions.to_tuple(False)
        self.transforms = A.Compose(
            [
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

    def generate_raw_images(self, filename):
        idx = int(filename.split(".")[0])
        nucleus_image = normalize_array(self.data[idx], None)  # C, H, W
        nucleus_image = np.moveaxis(nucleus_image, 0, -1)  # H, W, C
        return DatasetOutput(
            input=nucleus_image, target_array=np.asarray([0, 1, 0])
        )
