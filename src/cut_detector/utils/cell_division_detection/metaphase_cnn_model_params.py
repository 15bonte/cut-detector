from cnn_framework.utils.model_params.base_model_params import BaseModelParams
from cnn_framework.utils.dimensions import Dimensions


class MetaphaseCnnModelParams(BaseModelParams):
    """
    Metaphase CNN model params.
    """

    def __init__(self):
        super().__init__("metaphase_cnn")

        self.input_dimensions = Dimensions(height=256, width=256)

        self.num_epochs = 50
        self.learning_rate = 0.1

        self.nb_classes = 3
        self.class_names = ["Interphase", "Metaphase", "Death"]

        self.nb_modalities = 3
        self.nb_stacks_per_modality = 1
