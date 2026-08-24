import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from config import (
    BEST_MODEL_PATH,
    CLASS_NAMES,
    DEVICE,
    DROPOUT_RATE,
    IMAGE_SIZE,
    MODEL_NAME,
    NUM_CLASSES,
)
from model import create_model


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ResizeWithPadding:
    def __init__(self, size=224, fill=(0, 0, 0)):
        self.size = size
        self.fill = fill

    def __call__(self, image):
        image = image.convert("RGB")
        image.thumbnail((self.size, self.size), Image.Resampling.LANCZOS)
        new_image = Image.new("RGB", (self.size, self.size), self.fill)
        left = (self.size - image.width) // 2
        top = (self.size - image.height) // 2
        new_image.paste(image, (left, top))
        return new_image


def get_prediction_transform():
    return transforms.Compose(
        [
            ResizeWithPadding(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model():
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model bulunamadi: {BEST_MODEL_PATH}")

    device = torch.device(DEVICE)
    model = create_model(
        NUM_CLASSES,
        DROPOUT_RATE,
        pretrained=False,
        model_name=MODEL_NAME,
    )
    state_dict = load_state_dict(BEST_MODEL_PATH, device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_image(image_path, model=None):
    image_path = Path(image_path)

    if model is None:
        model = load_model()

    device = torch.device(DEVICE)
    transform = get_prediction_transform()

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities_tensor = torch.softmax(outputs, dim=1).squeeze(0).cpu()

    predicted_index = int(probabilities_tensor.argmax().item())
    confidence = float(probabilities_tensor[predicted_index].item())
    probabilities = {
        class_name: float(probabilities_tensor[index].item())
        for index, class_name in enumerate(CLASS_NAMES)
    }

    return {
        "predicted_index_0_based": predicted_index,
        "predicted_label_1_based": predicted_index + 1,
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": confidence,
        "probabilities": probabilities,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True, help="Tahmin edilecek gorsel yolu")
    args = parser.parse_args()

    result = predict_image(args.image_path)
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
