from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn


class MetaphaseCnn(nn.Module):
    def __init__(self, nb_classes, nb_input_channels=3):
        super().__init__()

        # Define Conv2d if input does not have 3 channels
        self.input_channels = nb_input_channels
        # input_channels, output_channels, kernel_size
        self.conv = nn.Conv2d(nb_input_channels, 3, 3)

        weights = ResNet50_Weights.DEFAULT
        self.cnn = resnet50(weights=weights)
        num_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Linear(num_features, nb_classes)

    def forward(self, x):
        if self.input_channels != 3:
            x = self.conv(x)
        return self.cnn(x)
