import torch

from cnn_framework.utils.data_loader_generators.classifier_data_loader_generator import (
    ClassifierDataLoaderGenerator,
)
from cnn_framework.utils.metrics.classification_accuracy import (
    ClassificationAccuracy,
)
from cnn_framework.utils.models.resnet_classifier import ResnetClassifier

from cut_detector.utils.mt_cut_detection.bridges_mt_model_manager import (
    BridgesMtModelManager,
)
from developers.training_and_evaluation.binary_bridges_cnn.bridges_parser import (
    BridgesParser,
)
from developers.training_and_evaluation.binary_bridges_cnn.data_set_inference import (
    BridgesCnnDataSet,
)
from developers.training_and_evaluation.binary_bridges_cnn.model_params import (
    BinaryBridgesModelParams,
)
from developers.training_and_evaluation.binary_bridges_cnn.tools import (
    evaluate_frame_error,
)


def main(params):
    loader_generator = ClassifierDataLoaderGenerator(params, BridgesCnnDataSet)
    _, _, test_dl = loader_generator.generate_data_loader()

    # Model definition
    model = ResnetClassifier(
        nb_classes=2,  # hard-coded from binary training
        nb_input_channels=len(params.c_indexes) * len(params.z_indexes),
        encoder_name=params.encoder_name,
    )
    model.load_state_dict(torch.load(params.model_load_path))

    manager = BridgesMtModelManager(model, params, ClassificationAccuracy)
    predictions = manager.predict(test_dl, nb_images_to_save=-1)
    manager.write_useful_information()

    evaluate_frame_error(
        predictions,
        test_dl.dataset.names,
        params.hmm_bridges_parameters_file,
    )


if __name__ == "__main__":
    parser = BridgesParser()
    args = parser.arguments_parser.parse_args()

    parameters = BinaryBridgesModelParams(inference_mode=True)
    parameters.update(args)

    main(parameters)
