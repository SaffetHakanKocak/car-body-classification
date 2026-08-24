import hashlib
import shutil
from pathlib import Path

from PIL import Image

from config import BASE_DIR


FORMULA_ONE_DIR = BASE_DIR / "external_datasets" / "formula-one-cars" / "Formula One Cars"
OPEN_WHEEL_SELECTED_DIR = BASE_DIR / "external_datasets" / "open_wheel_selected"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OUTPUT_PREFIX = "formula_one_open_wheel"
MIN_WIDTH = 250
MIN_HEIGHT = 150


def is_image_file(path):
    path = Path(path)
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def get_image_files(directory):
    directory = Path(directory)

    if not directory.exists():
        return []

    return sorted(path for path in directory.rglob("*") if is_image_file(path))


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


def collect_existing_hashes():
    existing_hashes = set()

    for image_path in get_image_files(OPEN_WHEEL_SELECTED_DIR):
        if is_valid_image(image_path):
            existing_hashes.add(file_hash(image_path))

    return existing_hashes


def next_available_index():
    index = 1

    while True:
        matches = list(OPEN_WHEEL_SELECTED_DIR.glob(f"{OUTPUT_PREFIX}_{index:06d}.*"))

        if not matches:
            return index

        index += 1


def main():
    print("Formula One gorselleri OPEN_WHEEL selected klasorune aktariliyor...")
    print(f"Kaynak klasor: {FORMULA_ONE_DIR}")
    print(f"Hedef klasor: {OPEN_WHEEL_SELECTED_DIR}")

    OPEN_WHEEL_SELECTED_DIR.mkdir(parents=True, exist_ok=True)

    source_files = get_image_files(FORMULA_ONE_DIR)
    existing_hashes = collect_existing_hashes()

    valid_count = 0
    duplicate_count = 0
    copied_count = 0
    target_index = next_available_index()

    for image_path in source_files:
        if not is_valid_image(image_path):
            continue

        valid_count += 1
        current_hash = file_hash(image_path)

        if current_hash in existing_hashes:
            duplicate_count += 1
            continue

        extension = image_path.suffix.lower()
        target_name = f"{OUTPUT_PREFIX}_{target_index:06d}{extension}"
        target_path = OPEN_WHEEL_SELECTED_DIR / target_name

        shutil.copy2(image_path, target_path)
        existing_hashes.add(current_hash)
        copied_count += 1
        target_index += 1

    final_count = len(get_image_files(OPEN_WHEEL_SELECTED_DIR))

    print()
    print(f"Formula One kaynak gorsel sayisi: {len(source_files)}")
    print(f"Gecerli kabul edilen gorsel sayisi: {valid_count}")
    print(f"Duplicate atlanan gorsel sayisi: {duplicate_count}")
    print(f"open_wheel_selected icine kopyalanan gorsel sayisi: {copied_count}")
    print(f"open_wheel_selected son toplam gorsel sayisi: {final_count}")
    print("Aktarma islemi Basarili.")


if __name__ == "__main__":
    main()
