from cnn_framework.utils.model_params.base_model_params import BaseModelParams
from cnn_framework.utils.dimensions import Dimensions


class BridgesMtCnnModelParams(BaseModelParams):
    """
    Bridges CNN model params.
    """

    def __init__(self):
        super().__init__("bridges_mt_cnn")

        self.input_dimensions = Dimensions(height=100, width=100)

        self.nb_classes = 3
        self.class_names = ["2 MT", "1 MT", "0 MT"]

        self.c_indexes = [0]
        self.z_indexes = [0]

        self.batch_size = 128

        self.encoder_name = "resnet18"

        self.num_epochs = 100
        self.learning_rate = 1e-4
