import torch

from cnn_framework.utils.model_managers.cnn_model_manager import (
    CnnModelManager,
)
from cnn_framework.utils.data_loader_generators.classifier_data_loader_generator import (
    ClassifierDataLoaderGenerator,
)
from cnn_framework.utils.metrics.classification_accuracy import (
    ClassificationAccuracy,
)
from cnn_framework.utils.models.resnet_classifier import ResnetClassifier

from developers.training_and_evaluation.binary_bridges_cnn.bridges_parser import (
    BridgesParser,
)
from developers.training_and_evaluation.binary_bridges_cnn.data_set import (
    BinaryBridgesCnnDataSet,
)
from developers.training_and_evaluation.binary_bridges_cnn.model_params import (
    BinaryBridgesModelParams,
)


def main(params):
    loader_generator = ClassifierDataLoaderGenerator(
        params, BinaryBridgesCnnDataSet
    )
    _, _, test_dl = loader_generator.generate_data_loader()

    # Model definition
    model = ResnetClassifier(
        nb_classes=2,  # hard-coded from binary training
        nb_input_channels=len(params.c_indexes) * len(params.z_indexes),
        encoder_name=params.encoder_name,
    )
    model.load_state_dict(torch.load(params.model_load_path))

    manager = CnnModelManager(model, params, ClassificationAccuracy)
    manager.predict(test_dl, nb_images_to_save=-1)

    manager.write_useful_information()


if __name__ == "__main__":
    parser = BridgesParser()
    args = parser.arguments_parser.parse_args()

    parameters = BinaryBridgesModelParams()
    parameters.update(args)

    main(parameters)
