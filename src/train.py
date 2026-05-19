import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    CLASS_NAMES,
    DEVICE,
    DROPOUT_RATE,
    IMAGE_SIZE,
    LEARNING_RATE,
    MODELS_DIR,
    NUM_CLASSES,
    NUM_EPOCHS,
    OUTPUTS_DIR,
    TRAIN_DIR,
    VAL_DIR,
)
from model import create_model


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
EARLY_STOPPING_PATIENCE = 5


class OrderedImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        class_to_idx = {class_name: index for index, class_name in enumerate(CLASS_NAMES)}
        return CLASS_NAMES, class_to_idx


def get_train_transform():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def create_dataloaders():
    train_dataset = OrderedImageFolder(TRAIN_DIR, transform=get_train_transform())
    val_dataset = OrderedImageFolder(VAL_DIR, transform=get_eval_transform())

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=DEVICE == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=DEVICE == "cuda",
    )

    return train_loader, val_loader


def run_train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        predictions = outputs.argmax(dim=1)
        correct_count += (predictions == labels).sum().item()
        sample_count += batch_size

    avg_loss = total_loss / sample_count
    accuracy = correct_count / sample_count
    return avg_loss, accuracy


def run_eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_count = 0
    sample_count = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            predictions = outputs.argmax(dim=1)
            correct_count += (predictions == labels).sum().item()
            sample_count += batch_size

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    avg_loss = total_loss / sample_count
    accuracy = correct_count / sample_count
    macro_f1 = float(
        f1_score(all_labels, all_predictions, average="macro", zero_division=0)
    )
    return avg_loss, accuracy, macro_f1


def save_history(history):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    history_path = OUTPUTS_DIR / "training_history.json"

    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)


def save_curves(history):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    epochs = [item["epoch"] for item in history]

    plt.figure()
    plt.plot(epochs, [item["train_loss"] for item in history], label="Train loss")
    plt.plot(epochs, [item["val_loss"] for item in history], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "loss_curve.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(epochs, [item["train_accuracy"] for item in history], label="Train accuracy")
    plt.plot(epochs, [item["val_accuracy"] for item in history], label="Val accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "accuracy_curve.png", dpi=150)
    plt.close()


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(DEVICE)
    print(f"Device: {device}")
    print(f"Sinif sirasi: {CLASS_NAMES}")

    train_loader, val_loader = create_dataloaders()
    print(f"Train gorsel sayisi: {len(train_loader.dataset)}")
    print(f"Val gorsel sayisi: {len(val_loader.dataset)}")

    model = create_model(
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE,
        pretrained=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_f1 = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_accuracy = run_train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_accuracy, val_macro_f1 = run_eval_epoch(
            model, val_loader, criterion, device
        )

        scheduler.step(val_macro_f1)

        epoch_info = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_macro_f1,
        }
        history.append(epoch_info)

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Train acc: {train_accuracy:.4f} | "
            f"Val acc: {val_accuracy:.4f} | "
            f"Val macro F1: {val_macro_f1:.4f}"
        )

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"En iyi model kaydedildi: {BEST_MODEL_PATH}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("Early stopping devreye girdi.")
            break

    save_history(history)
    save_curves(history)

    if Path(BEST_MODEL_PATH).exists():
        model_size_mb = BEST_MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"Best model dosya boyutu: {model_size_mb:.2f} MB")

        if model_size_mb > 95:
            print("Uyari: Model dosyasi 95 MB sinirini asiyor.")

    print(f"En iyi validation macro F1: {best_val_f1:.4f}")
    print("Egitim tamamlandi.")


if __name__ == "__main__":
    main()
