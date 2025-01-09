import albumentations as A
import numpy as np

from cnn_framework.utils.data_sets.abstract_data_set import AbstractDataSet
from cnn_framework.utils.readers.images_reader import ImagesReader
from cnn_framework.utils.data_sets.dataset_output import DatasetOutput
from cnn_framework.utils.enum import ProjectMethods
from cnn_framework.utils.tools import read_categories_probability_from_name
from cnn_framework.utils.tools import to_one_hot
from cnn_framework.utils.readers.utils.projection import Projection
from cnn_framework.utils.file_name_encoder import FileNameEncoder

from developers.training_and_evaluation.binary_bridges_cnn.tools import (
    get_category_from_name,
)


class BridgesCnnDataSet(AbstractDataSet):
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
                        axis=1,
                    )
                ]
            ],
        )  # SiRTubulin, MKLP1, PC

    def set_transforms(self):
        height, width = self.params.input_dimensions.to_tuple(False)
        if self.is_train:
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
                    A.Rotate(border_mode=0),
                    A.HorizontalFlip(),
                    A.VerticalFlip(),
                ]
            )
        else:
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

    def read_output(self, filename, one_hot=False):
        """
        Used for classification models.
        Read category and/or probabilities from file name
        """
        if self.params.mode == "microtubules":
            assert self.params.nb_classes == 3
        elif self.params.mode == "membrane":
            assert self.params.nb_classes == 2
        else:
            raise ValueError(f"Unknown mode: {self.params.mode}")

        # Read category from name
        categories_and_probabilities = filename.split(".")[0].split("_c")[1:]
        initial_category = int(categories_and_probabilities[0])

        # Match uncertain classes to basic ones
        category = get_category_from_name(
            initial_category, mode=self.params.mode
        )

        # Are there probabilities for classes?
        if len(categories_and_probabilities) > 1 and not one_hot:
            _, probabilities = read_categories_probability_from_name(filename)
        else:
            probabilities = to_one_hot(category, self.params.nb_classes)

        # Length of probabilities has to be equal to number of classes
        assert self.params.nb_classes == len(probabilities)

        # Category has to be the index of the highest probability
        assert category == np.argmax(probabilities)

        return np.asarray(probabilities)

    def generate_images(self, filename):
        # Output
        probabilities = self.read_output(filename)

        encoder = FileNameEncoder(
            self.params.names_train,
            self.params.names_val,
            self.params.names_test,
        )
        return DatasetOutput(
            input=self.input_data_source.get_image(filename),
            target_array=probabilities,
            additional=self.additional_data_source.get_image(filename),
            encoded_file_name=encoder.encode(filename),
        )
