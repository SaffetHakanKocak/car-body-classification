import json

from config import CLASS_NAMES, CLASS_NAMES_PATH, MODELS_DIR


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with CLASS_NAMES_PATH.open("w", encoding="utf-8") as file:
        json.dump(CLASS_NAMES, file, ensure_ascii=False, indent=4)

    print(f"Sinif isimleri kaydedildi: {CLASS_NAMES_PATH}")
    print(f"Sinif sayisi: {len(CLASS_NAMES)}")
    print("Sinif isimleri:")

    for index, class_name in enumerate(CLASS_NAMES):
        print(f"{index}: {class_name}")


if __name__ == "__main__":
    main()
