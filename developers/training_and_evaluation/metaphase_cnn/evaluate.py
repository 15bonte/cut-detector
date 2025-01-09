import torch

from cnn_framework.utils.data_loader_generators.classifier_data_loader_generator import (
    ClassifierDataLoaderGenerator,
)
from cnn_framework.utils.model_managers.cnn_model_manager import (
    CnnModelManager,
)
from cnn_framework.utils.data_managers.default_data_manager import (
    DefaultDataManager,
)
from cnn_framework.utils.metrics.classification_accuracy import (
    ClassificationAccuracy,
)
from cnn_framework.utils.parsers.cnn_parser import CnnParser
from cut_detector.utils.mitosis_track_generation.metaphase_cnn_model import (
    MetaphaseCnnModel,
)
from cut_detector.utils.mitosis_track_generation.metaphase_cnn_model_params import (
    MetaphaseCnnModelParams,
)
from developers.training_and_evaluation.metaphase_cnn.data_set import (
    MetaphaseCnnDataSet,
)


def main(params):
    loader_generator = ClassifierDataLoaderGenerator(
        params, MetaphaseCnnDataSet, DefaultDataManager
    )
    _, _, test_dl = loader_generator.generate_data_loader()

    # Model definition
    # Load pretrained model
    model = MetaphaseCnnModel(
        nb_classes=params.nb_classes,
        nb_input_channels=len(params.c_indexes) * len(params.z_indexes),
        encoder_name=params.encoder_name,
    )
    model.load_state_dict(torch.load(params.model_load_path))

    manager = CnnModelManager(model, params, ClassificationAccuracy)

    manager.predict(test_dl)

    manager.write_useful_information()


if __name__ == "__main__":
    parser = CnnParser()
    args = parser.arguments_parser.parse_args()

    parameters = MetaphaseCnnModelParams()
    parameters.update(args)

    main(parameters)
