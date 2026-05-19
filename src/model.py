import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def create_model(num_classes: int, dropout_rate: float = 0.3, pretrained: bool = True):
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes),
    )

    return model
