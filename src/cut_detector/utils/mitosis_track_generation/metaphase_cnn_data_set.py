import numpy as np

from cnn_framework.utils.data_sets.dataset_output import DatasetOutput
from cnn_framework.utils.preprocessing import normalize_array

from ..cnn_data_set import CnnDataSet


class MetaphaseCnnDataSet(CnnDataSet):
    """
    Only difference with parent class is normalization which is applied for
    each image independently.
    This is not something to be done, but current old model was developed that way.
    """

    def generate_images(self, filename: str) -> DatasetOutput:
        """Generate images from the given filename."""
        idx = int(filename.split(".")[0])
        nucleus_image = normalize_array(self.data[idx], None)  # CYX
        nucleus_image = np.moveaxis(nucleus_image, 0, -1)  # YXC
        # Define any target value
        target_array = np.zeros(self.params.nb_classes)
        target_array[0] = 1
        # Construct output
        return DatasetOutput(
            input=nucleus_image,
            target_array=target_array,
        )
