import torch
import torch.nn as nn
from torch import optim

from cnn_framework.utils.data_loader_generators.classifier_data_loader_generator import (
    ClassifierDataLoaderGenerator,
)
from cnn_framework.utils.model_managers.cnn_model_manager import CnnModelManager
from cnn_framework.utils.metrics.classification_accuracy import ClassificationAccuracy
from cnn_framework.utils.parsers.cnn_parser import CnnParser
from cnn_framework.utils.models.resnet_classifier import ResnetClassifier

from binary_bridges_cnn.data_set import BinaryBridgesCnnDataSet
from binary_bridges_cnn.model_params import BinaryBridgesModelParams


def main(params):
    loader_generator = ClassifierDataLoaderGenerator(params, BinaryBridgesCnnDataSet)
    train_dl, val_dl, test_dl = loader_generator.generate_data_loader()

    # Load pretrained model
    model = ResnetClassifier(
        nb_classes=params.nb_classes,  # output classes
        nb_input_channels=len(params.c_indexes) * len(params.z_indexes),
        encoder_name=params.encoder_name,
    )

    manager = CnnModelManager(model, params, ClassificationAccuracy)

    optimizer = optim.Adam(
        model.parameters(),
        lr=float(params.learning_rate),
        betas=(params.beta1, params.beta2),
    )  # define the optimization
    loss_function = nn.CrossEntropyLoss()  # define the loss function
    manager.fit(train_dl, val_dl, optimizer, loss_function)

    scores = []
    for model_path, name in zip(
        [manager.model_save_path_early_stopping, manager.model_save_path],
        ["early stopping", "final"],
    ):
        print(f"\nPredicting with {name} model.")
        # Update model with saved one
        manager.model.load_state_dict(torch.load(model_path))
        manager.predict(test_dl)
        score = manager.training_information.score
        scores.append(score)

    manager.training_information.score = max(scores)
    manager.write_useful_information()


if __name__ == "__main__":
    parser = CnnParser()
    args = parser.arguments_parser.parse_args()

    parameters = BinaryBridgesModelParams()
    parameters.update(args)

    main(parameters)
