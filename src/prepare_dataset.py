from config import CLASS_NAMES, RAW_DATA_DIR, TRAIN_DIR, VAL_DIR
from dataset_utils import (
    count_images_per_class,
    find_corrupted_images,
    split_dataset,
)


def print_class_counts(title, counts):
    print(title)

    for class_name in CLASS_NAMES:
        print(f"{class_name}: {counts[class_name]}")


def main():
    print("Raw veri seti kontrol ediliyor...")
    raw_counts = count_images_per_class(RAW_DATA_DIR, CLASS_NAMES)
    print_class_counts("Raw sinif gorsel sayilari:", raw_counts)

    print()
    print("Bozuk gorseller kontrol ediliyor...")
    corrupted_images = find_corrupted_images(RAW_DATA_DIR, CLASS_NAMES)

    if corrupted_images:
        print(f"Uyari: {len(corrupted_images)} bozuk veya okunamayan gorsel bulundu.")
        print("Bu dosyalar silinmedi. Split sirasinda kopyalanmayacaklar:")

        for image_path in corrupted_images:
            print(image_path)
    else:
        print("Bozuk gorsel bulunmadi.")

    print()
    print("Train/val split islemi baslatiliyor...")
    split_counts = split_dataset(
        RAW_DATA_DIR,
        TRAIN_DIR,
        VAL_DIR,
        CLASS_NAMES,
        val_ratio=0.2,
        seed=42,
    )

    print()
    print_class_counts("Train sinif gorsel sayilari:", split_counts["train"])

    print()
    print_class_counts("Val sinif gorsel sayilari:", split_counts["val"])

    total_train = sum(split_counts["train"].values())
    total_val = sum(split_counts["val"].values())

    print()
    print(f"Toplam train gorsel sayisi: {total_train}")
    print(f"Toplam val gorsel sayisi: {total_val}")
    print("Dataset hazirlama tamamlandi.")


if __name__ == "__main__":
    main()
