import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import (
    BEST_MODEL_PATH,
    CLASS_NAMES,
    DEVICE,
    DROPOUT_RATE,
    IMAGE_SIZE,
    NUM_CLASSES,
    OUTPUTS_DIR,
    TEST_DIR,
)
from model import create_model


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class OrderedImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        class_to_idx = {class_name: index for index, class_name in enumerate(CLASS_NAMES)}
        return CLASS_NAMES, class_to_idx


def get_eval_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(device):
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model bulunamadi: {BEST_MODEL_PATH}")

    model = create_model(NUM_CLASSES, DROPOUT_RATE, pretrained=False)
    state_dict = load_state_dict(BEST_MODEL_PATH, device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def collect_predictions(model, dataloader, device):
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.tolist())
            all_predictions.extend(predictions.cpu().tolist())

    return all_labels, all_predictions


def save_confusion_matrix(y_true, y_pred):
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        normalize="true",
    )
    matrix = np.nan_to_num(matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.xlabel("Tahmin sinifi")
    plt.ylabel("Gercek sinif")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "normalized_confusion_matrix.png", dpi=150)
    plt.close()


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(DEVICE)
    print(f"Device: {device}")

    test_dataset = OrderedImageFolder(TEST_DIR, transform=get_eval_transform())
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"Test gorsel sayisi: {len(test_dataset)}")

    model = load_model(device)
    y_true, y_pred = collect_predictions(model, test_loader, device)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "weighted_recall": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }

    (OUTPUTS_DIR / "test_classification_report.txt").write_text(
        report_text, encoding="utf-8"
    )

    with (OUTPUTS_DIR / "test_classification_report.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report_dict, file, indent=4)

    with (OUTPUTS_DIR / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    save_confusion_matrix(y_true, y_pred)

    print("Test metrikleri:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro precision: {metrics['macro_precision']:.4f}")
    print(f"Macro recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print("Degerlendirme tamamlandi.")


if __name__ == "__main__":
    main()
