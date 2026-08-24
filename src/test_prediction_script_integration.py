import shutil
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
SUBMISSION_DIR = BASE_DIR / "230202058_230202050"
TEST_DIR = BASE_DIR / "dataset" / "processed" / "test"
SAMPLE_DIR = BASE_DIR / "testdata_sample"

CLASS_NAMES = [
    "SUV",
    "VAN",
    "STATION_WAGON",
    "MICRO",
    "OPEN_WHEEL",
    "SEDAN",
    "HATCHBACK",
    "PICK_UP",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGES_PER_CLASS = 2


def check_required_files():
    required_files = [
        SUBMISSION_DIR / "PredictionScript.txt",
        SUBMISSION_DIR / "best_model.pth",
        SUBMISSION_DIR / "class_names.json",
    ]

    print("Submission file check:")

    all_ok = True
    for file_path in required_files:
        exists = file_path.exists()
        status = "OK" if exists else "MISSING"
        print(f"{status}: {file_path}")
        all_ok = all_ok and exists

    if not all_ok:
        raise FileNotFoundError("Submission folder has missing files.")


def get_class_images(class_dir):
    image_paths = [
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(image_paths, key=lambda path: (path.name.lower(), str(path).lower()))


def create_testdata_sample():
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Processed test directory not found: {TEST_DIR}")

    if SAMPLE_DIR.exists():
        shutil.rmtree(SAMPLE_DIR)

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("Creating testdata_sample:")

    total_copied = 0

    for label_index, class_name in enumerate(CLASS_NAMES, start=1):
        source_dir = TEST_DIR / class_name
        target_dir = SAMPLE_DIR / str(label_index)
        target_dir.mkdir(parents=True, exist_ok=True)

        if not source_dir.exists():
            print(f"Warning: class folder not found: {source_dir}")
            continue

        selected_images = get_class_images(source_dir)[:IMAGES_PER_CLASS]

        for image_path in selected_images:
            shutil.copy2(image_path, target_dir / image_path.name)
            total_copied += 1

        print(f"{label_index} ({class_name}): {len(selected_images)} images copied")

    print(f"Total sample images copied: {total_copied}")
    print(f"Sample folder: {SAMPLE_DIR}")


def print_colab_instructions():
    print()
    print("Colab integration test steps:")
    print("1. Upload 230202058_230202050 folder to Colab.")
    print("2. Upload testdata_sample folder to /content/testdata_sample.")
    print("3. Run these commands in Colab:")
    print('   exec(open("PredictionScript.txt", encoding="utf-8").read())')
    print('   Predict("/content/testdata_sample")')
    print("4. Check that Preds.txt is created.")


def main():
    check_required_files()
    create_testdata_sample()
    print_colab_instructions()
    print()
    print("Integration preparation completed.")


if __name__ == "__main__":
    main()
