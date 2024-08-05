import albumentations as A

from cnn_framework.utils.data_sets.abstract_data_set import AbstractDataSet
from cnn_framework.utils.readers.images_reader import ImagesReader
from cnn_framework.utils.data_sets.dataset_output import DatasetOutput
from cnn_framework.utils.enum import ProjectMethods
from cnn_framework.utils.readers.utils.projection import Projection
from cnn_framework.utils.preprocessing import normalize_array


class MetaphaseCnnDataSet(AbstractDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Data sources
        self.input_data_source = ImagesReader(
            [self.data_manager.get_microscopy_image_path],
            [
                [
                    Projection(
                        method=ProjectMethods.Channel,
                        channels=self.params.c_indexes,
                        axis=-1,
                    ),
                ]
            ],
        )

    def set_transforms(self):
        height, width = self.params.input_dimensions.to_tuple(False)
        if self.is_train:
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
                    A.Rotate(border_mode=0),
                    A.HorizontalFlip(),
                    A.VerticalFlip(),
                ]
            )
        else:
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

    def generate_images(self, filename):
        # Output
        probabilities = self.read_output(filename)

        input_image = self.input_data_source.get_image(filename)
        input_image = normalize_array(input_image, None)  # YXC

        return DatasetOutput(
            input=input_image,
            target_array=probabilities,
        )
