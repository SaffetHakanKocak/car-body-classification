import random
import shutil
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(path):
    path = Path(path)
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def get_image_files(class_dir):
    class_dir = Path(class_dir)

    if not class_dir.exists():
        return []

    image_files = [path for path in class_dir.iterdir() if is_image_file(path)]
    return sorted(image_files)


def check_image_validity(image_path):
    image_path = Path(image_path)

    try:
        with Image.open(image_path) as image:
            image.verify()

        with Image.open(image_path) as image:
            image.load()

        return True
    except Exception:
        return False


def count_images_per_class(raw_data_dir, class_names):
    raw_data_dir = Path(raw_data_dir)
    counts = {}

    for class_name in class_names:
        class_dir = raw_data_dir / class_name
        counts[class_name] = len(get_image_files(class_dir))

    return counts


def find_corrupted_images(raw_data_dir, class_names):
    raw_data_dir = Path(raw_data_dir)
    corrupted_images = []

    for class_name in class_names:
        class_dir = raw_data_dir / class_name

        for image_path in get_image_files(class_dir):
            if not check_image_validity(image_path):
                corrupted_images.append(image_path)

    return corrupted_images


def clear_directory(directory):
    directory = Path(directory)

    if directory.exists():
        shutil.rmtree(directory)

    directory.mkdir(parents=True, exist_ok=True)


def create_class_folders(base_dir, class_names):
    base_dir = Path(base_dir)

    for class_name in class_names:
        (base_dir / class_name).mkdir(parents=True, exist_ok=True)


def split_dataset(raw_data_dir, train_dir, val_dir, class_names, val_ratio=0.2, seed=42):
    raw_data_dir = Path(raw_data_dir)
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)

    clear_directory(train_dir)
    clear_directory(val_dir)
    create_class_folders(train_dir, class_names)
    create_class_folders(val_dir, class_names)

    rng = random.Random(seed)
    split_counts = {
        "train": {class_name: 0 for class_name in class_names},
        "val": {class_name: 0 for class_name in class_names},
    }

    for class_name in class_names:
        class_dir = raw_data_dir / class_name
        image_files = [
            image_path
            for image_path in get_image_files(class_dir)
            if check_image_validity(image_path)
        ]

        if len(image_files) == 0:
            print(f"Uyari: {class_name} sinifinda gecerli gorsel yok, devam ediliyor.")
            continue

        rng.shuffle(image_files)

        if len(image_files) == 1:
            val_count = 0
        else:
            val_count = max(1, int(len(image_files) * val_ratio))
            val_count = min(val_count, len(image_files) - 1)

        val_files = image_files[:val_count]
        train_files = image_files[val_count:]

        for image_path in train_files:
            target_path = train_dir / class_name / image_path.name
            shutil.copy2(image_path, target_path)

        for image_path in val_files:
            target_path = val_dir / class_name / image_path.name
            shutil.copy2(image_path, target_path)

        split_counts["train"][class_name] = len(train_files)
        split_counts["val"][class_name] = len(val_files)

    return split_counts
