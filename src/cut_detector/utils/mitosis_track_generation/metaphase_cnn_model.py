import torch.nn as nn

from cnn_framework.utils.models.resnet_classifier import ResnetClassifier


class MetaphaseCnnModel(ResnetClassifier):
    """Model for metaphase classification."""

    def __init__(
        self,
        nb_classes: int,
        nb_input_channels: int,
        encoder_name: str,
    ):
        super().__init__(
            nb_classes=nb_classes,
            nb_input_channels=nb_input_channels,
            encoder_name=encoder_name,
        )
        # NB: following is totally unused, but old model defines it
        self.conv = nn.Conv2d(nb_input_channels, 3, 3)
