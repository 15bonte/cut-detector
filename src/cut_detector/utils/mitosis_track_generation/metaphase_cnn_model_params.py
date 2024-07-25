from cnn_framework.utils.model_params.base_model_params import BaseModelParams
from cnn_framework.utils.dimensions import Dimensions


class MetaphaseCnnModelParams(BaseModelParams):
    """Metaphase CNN model params."""

    def __init__(self):
        super().__init__("metaphase_cnn")

        self.input_dimensions = Dimensions(height=256, width=256)

        self.nb_classes = 3
        self.class_names = ["Interphase", "Metaphase", "Death"]

        self.c_indexes = [0, 1, 2]
        self.z_indexes = [0]

        self.batch_size = 128

        self.encoder_name = "resnet50"

        self.num_epochs = 500
        self.learning_rate = 5e-5
