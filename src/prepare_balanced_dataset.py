import random
import shutil

from config import CLASS_NAMES, RAW_DATA_DIR, TEST_DIR, TRAIN_DIR, VAL_DIR
from dataset_utils import (
    check_image_validity,
    clear_directory,
    create_class_folders,
    get_image_files,
)


MAX_IMAGES_PER_CLASS = 1000
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
SEED = 42


def print_class_counts(title, counts):
    print(title)

    for class_name in CLASS_NAMES:
        print(f"{class_name}: {counts[class_name]}")


def prepare_output_dirs():
    clear_directory(TRAIN_DIR)
    clear_directory(VAL_DIR)
    clear_directory(TEST_DIR)

    create_class_folders(TRAIN_DIR, CLASS_NAMES)
    create_class_folders(VAL_DIR, CLASS_NAMES)
    create_class_folders(TEST_DIR, CLASS_NAMES)


def copy_files(image_files, target_dir, class_name):
    for image_path in image_files:
        target_path = target_dir / class_name / image_path.name
        shutil.copy2(image_path, target_path)


def main():
    print("Dengeli train/val/test veri seti hazirlaniyor...")
    print("Raw klasoru degistirilmeyecek, sadece processed klasoru yenilenecek.")

    prepare_output_dirs()
    rng = random.Random(SEED)

    raw_counts = {}
    selected_counts = {}
    train_counts = {}
    val_counts = {}
    test_counts = {}

    for class_name in CLASS_NAMES:
        class_dir = RAW_DATA_DIR / class_name
        raw_files = get_image_files(class_dir)
        valid_files = [
            image_path
            for image_path in raw_files
            if check_image_validity(image_path)
        ]

        raw_counts[class_name] = len(raw_files)

        if len(valid_files) == 0:
            print(f"Uyari: {class_name} sinifinda gecerli gorsel yok.")
            selected_files = []
        else:
            rng.shuffle(valid_files)
            selected_files = valid_files[:MAX_IMAGES_PER_CLASS]

        selected_count = len(selected_files)
        train_count = int(selected_count * TRAIN_RATIO)
        val_count = int(selected_count * VAL_RATIO)
        test_count = selected_count - train_count - val_count

        train_files = selected_files[:train_count]
        val_files = selected_files[train_count : train_count + val_count]
        test_files = selected_files[train_count + val_count :]

        copy_files(train_files, TRAIN_DIR, class_name)
        copy_files(val_files, VAL_DIR, class_name)
        copy_files(test_files, TEST_DIR, class_name)

        selected_counts[class_name] = selected_count
        train_counts[class_name] = len(train_files)
        val_counts[class_name] = len(val_files)
        test_counts[class_name] = len(test_files)

    print()
    print_class_counts("Raw sinif gorsel sayilari:", raw_counts)

    print()
    print_class_counts("Secilen sinif gorsel sayilari:", selected_counts)

    print()
    print_class_counts("Train sinif gorsel sayilari:", train_counts)

    print()
    print_class_counts("Val sinif gorsel sayilari:", val_counts)

    print()
    print_class_counts("Test sinif gorsel sayilari:", test_counts)

    print()
    print(f"Toplam train gorsel sayisi: {sum(train_counts.values())}")
    print(f"Toplam val gorsel sayisi: {sum(val_counts.values())}")
    print(f"Toplam test gorsel sayisi: {sum(test_counts.values())}")
    print("Dengeli veri seti hazirlama Basarili.")


if __name__ == "__main__":
    main()
