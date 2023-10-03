import albumentations as A

from cnn_framework.utils.data_sets.abstract_data_set import AbstractDataSet
from cnn_framework.utils.readers.images_reader import ImagesReader
from cnn_framework.utils.data_sets.dataset_output import DatasetOutput
from cnn_framework.utils.enum import NormalizeMethods, ProjectMethods
from cnn_framework.utils.readers.utils.projection import Projection
from cnn_framework.utils.readers.utils.normalization import Normalization


class MetaphaseCnnDataSet(AbstractDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Data sources
        nb_channels = self.params.nb_modalities * self.params.nb_stacks_per_modality
        self.input_data_source = ImagesReader(
            [self.data_manager.get_microscopy_image_path],
            [[Projection(method=ProjectMethods.Channel, channels=list(range(nb_channels)))]],
            [Normalization(method=NormalizeMethods.Standardize)],
        )

    def set_transforms(self):
        height, width = self.params.input_dimensions.to_tuple(False)
        if self.is_train:
            self.transforms = A.Compose(
                [
                    A.PadIfNeeded(min_height=height, min_width=width, border_mode=0, value=0, p=1),
                    A.CenterCrop(height=height, width=width, p=1),
                    A.Rotate(border_mode=0),
                    A.HorizontalFlip(),
                    A.VerticalFlip(),
                ]
            )
        else:
            self.transforms = A.Compose(
                [
                    A.PadIfNeeded(min_height=height, min_width=width, border_mode=0, value=0, p=1),
                    A.CenterCrop(height=height, width=width, p=1),
                ]
            )

    def generate_raw_images(self, filename):
        # Output
        probabilities = self.read_output(filename)

        return DatasetOutput(
            input=self.input_data_source.get_image(filename),
            target_array=probabilities,
            additional=self.additional_data_source.get_image(filename),
        )
