import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR / "src"))

from config import (
    BEST_MODEL_PATH,
    CLASS_NAMES,
    DEVICE,
    DROPOUT_RATE,
    IMAGE_SIZE,
    MODEL_NAME,
    NUM_CLASSES,
    OUTPUTS_DIR,
    TEST_DIR,
)
from dataset_utils import IMAGE_EXTENSIONS
from model import create_model


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

ANALYSIS_CSV_PATH = OUTPUTS_DIR / "test_prediction_analysis.csv"
WRONG_CSV_PATH = OUTPUTS_DIR / "wrong_predictions.csv"
CORRECT_CSV_PATH = OUTPUTS_DIR / "correct_predictions.csv"
WRONG_IMAGES_DIR = OUTPUTS_DIR / "wrong_predictions_images"
CORRECT_IMAGES_DIR = OUTPUTS_DIR / "correct_predictions_images"


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


def get_transform():
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


def load_model(device):
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model bulunamadi: {BEST_MODEL_PATH}")

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


def get_test_images():
    image_paths = [
        path
        for path in TEST_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(image_paths, key=lambda path: str(path).lower())


def predict_image(image_path, model, transform, device):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze(0).cpu()

    predicted_index = int(probabilities.argmax().item())
    confidence = float(probabilities[predicted_index].item())
    return predicted_index, confidence


def safe_filename_part(text):
    return "".join(char if char.isalnum() or char in ("_", "-", ".") else "_" for char in text)


def clear_directory(directory):
    if directory.exists():
        shutil.rmtree(directory)

    directory.mkdir(parents=True, exist_ok=True)


def copy_analysis_image(row, target_dir):
    image_path = Path(row["image_path"])
    confidence_text = f"{row['confidence']:.2f}"
    file_name = (
        f"{row['true_class']}__pred_{row['predicted_class']}"
        f"__confidence_{confidence_text}__{image_path.name}"
    )
    safe_name = safe_filename_part(file_name)
    shutil.copy2(image_path, target_dir / safe_name)


def analyze_predictions(copy_correct=False):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    clear_directory(WRONG_IMAGES_DIR)

    if copy_correct:
        clear_directory(CORRECT_IMAGES_DIR)

    device = torch.device(DEVICE)
    model = load_model(device)
    transform = get_transform()
    image_paths = get_test_images()
    rows = []

    for image_path in image_paths:
        true_class = image_path.parent.name

        if true_class not in CLASS_NAMES:
            print(f"Uyari: Bilinmeyen sinif klasoru atlandi: {image_path}")
            continue

        true_index = CLASS_NAMES.index(true_class)
        predicted_index, confidence = predict_image(image_path, model, transform, device)
        predicted_class = CLASS_NAMES[predicted_index]
        is_correct = true_index == predicted_index

        rows.append(
            {
                "file_name": image_path.name,
                "image_path": str(image_path),
                "true_class": true_class,
                "true_label_1_based": true_index + 1,
                "predicted_class": predicted_class,
                "predicted_label_1_based": predicted_index + 1,
                "confidence": confidence,
                "is_correct": is_correct,
            }
        )

    return rows


def save_outputs(rows, copy_correct=False):
    df = pd.DataFrame(rows)
    wrong_df = df[df["is_correct"] == False].copy()
    correct_df = df[df["is_correct"] == True].copy()

    df.to_csv(ANALYSIS_CSV_PATH, index=False, encoding="utf-8")
    wrong_df.to_csv(WRONG_CSV_PATH, index=False, encoding="utf-8")
    correct_df.to_csv(CORRECT_CSV_PATH, index=False, encoding="utf-8")

    for _, row in wrong_df.iterrows():
        copy_analysis_image(row, WRONG_IMAGES_DIR)

    if copy_correct:
        for _, row in correct_df.iterrows():
            copy_analysis_image(row, CORRECT_IMAGES_DIR)

    return df, wrong_df, correct_df


def print_summary(df, wrong_df, correct_df):
    total_count = len(df)
    correct_count = len(correct_df)
    wrong_count = len(wrong_df)
    accuracy = correct_count / total_count if total_count > 0 else 0.0

    print(f"Toplam test gorsel sayisi: {total_count}")
    print(f"Dogru tahmin sayisi: {correct_count}")
    print(f"Yanlis tahmin sayisi: {wrong_count}")
    print(f"Accuracy: {accuracy:.4f}")

    print()
    print("Sinif bazinda yanlis tahmin sayilari:")
    wrong_by_class = Counter(wrong_df["true_class"].tolist())

    for class_name in CLASS_NAMES:
        print(f"{class_name}: {wrong_by_class.get(class_name, 0)}")

    print()
    print("En cok karisan sinif ciftleri:")
    pair_counts = Counter(
        zip(wrong_df["true_class"].tolist(), wrong_df["predicted_class"].tolist())
    )

    if not pair_counts:
        print("Yanlis tahmin yok.")
    else:
        for (true_class, predicted_class), count in pair_counts.most_common(20):
            print(f"{true_class} -> {predicted_class}: {count} gorsel")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--copy_correct",
        action="store_true",
        help="Dogru tahmin edilen gorselleri de outputs/correct_predictions_images icine kopyalar.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = analyze_predictions(copy_correct=args.copy_correct)
    df, wrong_df, correct_df = save_outputs(rows, copy_correct=args.copy_correct)
    print_summary(df, wrong_df, correct_df)

    print()
    print(f"Analiz CSV: {ANALYSIS_CSV_PATH}")
    print(f"Yanlis tahmin CSV: {WRONG_CSV_PATH}")
    print(f"Dogru tahmin CSV: {CORRECT_CSV_PATH}")
    print(f"Yanlis gorsel klasoru: {WRONG_IMAGES_DIR}")

    if args.copy_correct:
        print(f"Dogru gorsel klasoru: {CORRECT_IMAGES_DIR}")


if __name__ == "__main__":
    main()
