import shutil
from pathlib import Path

from config import BASE_DIR, CLASS_NAMES, RAW_DATA_DIR
from dataset_utils import (
    check_image_validity,
    count_images_per_class,
    create_class_folders,
    get_image_files,
)


CARS_CROPPED_DIR = (
    BASE_DIR / "external_datasets" / "cars-body-type-cropped" / "Cars_Body_Type"
)
STANFORD_DIR = (
    BASE_DIR / "external_datasets" / "stanford-car-body-type-data" / "stanford_cars_type"
)

CARS_CROPPED_SPLITS = ["train", "valid", "test"]

DATASET_CONFIGS = [
    {
        "name": "cars-body-type-cropped",
        "prefix": "carscropped",
        "base_dir": CARS_CROPPED_DIR,
        "splits": CARS_CROPPED_SPLITS,
        "mapping": {
            "SUV": "SUV",
            "VAN": "VAN",
            "Sedan": "SEDAN",
            "Hatchback": "HATCHBACK",
            "Pick-Up": "PICK_UP",
        },
    },
    {
        "name": "stanford-car-body-type-data",
        "prefix": "stanford",
        "base_dir": STANFORD_DIR,
        "splits": None,
        "mapping": {
            "SUV": "SUV",
            "Van": "VAN",
            "Sedan": "SEDAN",
            "Hatchback": "HATCHBACK",
            "Wagon": "STATION_WAGON",
        },
    },
]

MISSING_SOURCE_CLASSES = ["MICRO", "OPEN_WHEEL"]


def collect_source_images(dataset_config):
    source_records = []

    for source_class, target_class in dataset_config["mapping"].items():
        image_paths = []

        if dataset_config["splits"] is None:
            source_dirs = [dataset_config["base_dir"] / source_class]
        else:
            source_dirs = [
                dataset_config["base_dir"] / split_name / source_class
                for split_name in dataset_config["splits"]
            ]

        for source_dir in source_dirs:
            image_paths.extend(get_image_files(source_dir))

        valid_images = []
        corrupted_count = 0

        for image_path in sorted(image_paths):
            if check_image_validity(image_path):
                valid_images.append(image_path)
            else:
                corrupted_count += 1

        source_records.append(
            {
                "dataset_name": dataset_config["name"],
                "prefix": dataset_config["prefix"],
                "source_class": source_class,
                "target_class": target_class,
                "valid_images": valid_images,
                "corrupted_count": corrupted_count,
            }
        )

    return source_records


def build_import_plan():
    import_plan = []

    for dataset_config in DATASET_CONFIGS:
        import_plan.extend(collect_source_images(dataset_config))

    return import_plan


def print_analysis_report(import_plan):
    print("Analiz raporu")
    print()
    print("Kaynak sinif gecerli gorsel sayilari:")

    for record in import_plan:
        valid_count = len(record["valid_images"])
        corrupted_count = record["corrupted_count"]
        dataset_name = record["dataset_name"]
        source_class = record["source_class"]
        target_class = record["target_class"]

        print(
            f"{dataset_name} / {source_class} -> {target_class}: "
            f"{valid_count} gecerli gorsel"
        )

        if corrupted_count > 0:
            print(f"Uyari: {corrupted_count} bozuk gorsel kopyalanmayacak.")

    target_counts = {class_name: 0 for class_name in CLASS_NAMES}

    for record in import_plan:
        target_counts[record["target_class"]] += len(record["valid_images"])

    print()
    print("Proje siniflarina kopyalanacak gorsel sayilari:")

    for class_name in CLASS_NAMES:
        print(f"{class_name}: {target_counts[class_name]}")

    print_missing_source_warnings()


def print_missing_source_warnings():
    print()

    for class_name in MISSING_SOURCE_CLASSES:
        print(
            f"Uyari: {class_name} sinifi icin bu iki Kaggle datasetinde "
            "dogrudan kaynak yok, ek veri toplanmali."
        )


def clear_previous_imports(raw_data_dir, class_names):
    prefixes = [dataset_config["prefix"] for dataset_config in DATASET_CONFIGS]
    removed_count = 0

    for class_name in class_names:
        class_dir = Path(raw_data_dir) / class_name

        if not class_dir.exists():
            continue

        for prefix in prefixes:
            for file_path in class_dir.glob(f"{prefix}_*"):
                if file_path.is_file():
                    file_path.unlink()
                    removed_count += 1

    return removed_count


def copy_images(import_plan, raw_data_dir):
    copied_counts = {class_name: 0 for class_name in CLASS_NAMES}

    for record in import_plan:
        prefix = record["prefix"]
        target_class = record["target_class"]
        target_dir = Path(raw_data_dir) / target_class

        for index, image_path in enumerate(record["valid_images"], start=1):
            extension = image_path.suffix.lower()
            new_name = f"{prefix}_{target_class}_{index:06d}{extension}"
            target_path = target_dir / new_name

            shutil.copy2(image_path, target_path)
            copied_counts[target_class] += 1

    return copied_counts


def print_counts(title, counts):
    print(title)

    for class_name in CLASS_NAMES:
        print(f"{class_name}: {counts[class_name]}")


def main():
    print("External dataset import islemi baslatiliyor...")
    create_class_folders(RAW_DATA_DIR, CLASS_NAMES)

    import_plan = build_import_plan()
    print_analysis_report(import_plan)

    print()
    print("Onceki carscropped_* ve stanford_* dosyalari temizleniyor...")
    removed_count = clear_previous_imports(RAW_DATA_DIR, CLASS_NAMES)
    print(f"Temizlenen dosya sayisi: {removed_count}")

    print()
    print("Gorseller dataset/raw klasorune kopyalaniyor...")
    copied_counts = copy_images(import_plan, RAW_DATA_DIR)
    print_counts("Kopyalanan gorsel sayilari:", copied_counts)

    print()
    final_counts = count_images_per_class(RAW_DATA_DIR, CLASS_NAMES)
    print_counts("dataset/raw son gorsel sayilari:", final_counts)

    print()
    print("External dataset import islemi Basarili.")


if __name__ == "__main__":
    main()
