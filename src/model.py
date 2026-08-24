import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B2_Weights,
    efficientnet_b0,
    efficientnet_b2,
)


def create_model(
    num_classes: int,
    dropout_rate: float = 0.4,
    pretrained: bool = True,
    model_name: str = "efficientnet_b2",
):
    model_name = model_name.lower()

    if model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
    elif model_name == "efficientnet_b2":
        weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
        model = efficientnet_b2(weights=weights)
    else:
        raise ValueError(f"Desteklenmeyen model adi: {model_name}")

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes),
    )

    return model
