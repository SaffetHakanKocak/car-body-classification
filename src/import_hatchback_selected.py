import hashlib
import shutil
from pathlib import Path

from PIL import Image

from config import BASE_DIR, RAW_DATA_DIR
from dataset_utils import get_image_files


HATCHBACK_SELECTED_DIR = BASE_DIR / "external_datasets" / "hatchback_selected"
HATCHBACK_RAW_DIR = RAW_DATA_DIR / "HATCHBACK"

MIN_WIDTH = 250
MIN_HEIGHT = 150
OUTPUT_PREFIX = "selected_hatchback"


def file_hash(file_path):
    hash_value = hashlib.sha256()

    with Path(file_path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hash_value.update(chunk)

    return hash_value.hexdigest()


def is_valid_image(image_path):
    try:
        with Image.open(image_path) as image:
            width, height = image.size
            image.verify()

        with Image.open(image_path) as image:
            image.load()

        return width >= MIN_WIDTH and height >= MIN_HEIGHT
    except Exception:
        return False


def clear_previous_selected_imports():
    HATCHBACK_RAW_DIR.mkdir(parents=True, exist_ok=True)
    removed_count = 0

    for file_path in HATCHBACK_RAW_DIR.glob(f"{OUTPUT_PREFIX}_*"):
        if file_path.is_file():
            file_path.unlink()
            removed_count += 1

    return removed_count


def count_hatchback_raw_images():
    return len(get_image_files(HATCHBACK_RAW_DIR))


def import_selected_images():
    HATCHBACK_SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    HATCHBACK_RAW_DIR.mkdir(parents=True, exist_ok=True)

    selected_files = get_image_files(HATCHBACK_SELECTED_DIR)
    valid_files = []
    known_hashes = set()
    duplicate_count = 0

    for image_path in selected_files:
        if not is_valid_image(image_path):
            continue

        current_hash = file_hash(image_path)

        if current_hash in known_hashes:
            duplicate_count += 1
            continue

        known_hashes.add(current_hash)
        valid_files.append(image_path)

    copied_count = 0

    for index, image_path in enumerate(valid_files, start=1):
        extension = image_path.suffix.lower()
        target_name = f"{OUTPUT_PREFIX}_{index:06d}{extension}"
        target_path = HATCHBACK_RAW_DIR / target_name

        shutil.copy2(image_path, target_path)
        copied_count += 1

    return {
        "found": len(selected_files),
        "valid": len(valid_files),
        "duplicates": duplicate_count,
        "copied": copied_count,
    }


def main():
    print("HATCHBACK secili gorsel import islemi baslatiliyor...")
    print(f"Kaynak klasor: {HATCHBACK_SELECTED_DIR}")
    print(f"Hedef klasor: {HATCHBACK_RAW_DIR}")

    removed_count = clear_previous_selected_imports()
    print(f"Onceki selected_hatchback_* dosya sayisi temizlendi: {removed_count}")

    stats = import_selected_images()

    print()
    print(f"hatchback_selected klasorunde bulunan dosya sayisi: {stats['found']}")
    print(f"Gecerli kabul edilen gorsel sayisi: {stats['valid']}")
    print(f"Duplicate atlanan gorsel sayisi: {stats['duplicates']}")
    print(f"dataset/raw/HATCHBACK icine kopyalanan gorsel sayisi: {stats['copied']}")
    print(f"dataset/raw/HATCHBACK son toplam gorsel sayisi: {count_hatchback_raw_images()}")
    print("HATCHBACK secili gorsel import islemi Basarili.")


if __name__ == "__main__":
    main()
