from cnn_framework.utils.model_params.base_model_params import BaseModelParams
from cnn_framework.utils.dimensions import Dimensions


class BinaryBridgesModelParams(BaseModelParams):
    """
    Bridges CNN model params.
    """

    def __init__(self, inference_mode=False):
        super().__init__("binary_bridges_cnn")

        self.input_dimensions = Dimensions(height=100, width=100)

        self.num_epochs = 100
        self.learning_rate = 1e-4
        self.c_indexes = [0]
        self.mode = "microtubules"
        self.encoder_name = "resnet18"
        self.num_epochs = 100
        self.learning_rate = 1e-4

        if inference_mode:
            self.class_names = ["2 MT", "1 MT", "0 MT"]
        else:
            self.class_names = ["No micro-tubules", "Micro-tubules"]

        self.nb_classes = len(self.class_names)
        self.hmm_bridges_parameters_file = ""

    def update(self, args=None):
        super().update(args)

        if args.hmm_bridges_parameters_file:
            self.hmm_bridges_parameters_file = args.hmm_bridges_parameters_file
