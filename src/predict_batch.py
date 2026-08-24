import argparse
from collections import Counter
from pathlib import Path

from config import TEST_DIR
from dataset_utils import IMAGE_EXTENSIONS
from predict import load_model, predict_image


def get_image_paths(input_dir):
    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input klasoru bulunamadi: {input_dir}")

    image_paths = [
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(image_paths, key=lambda path: (path.name.lower(), str(path).lower()))


def warn_duplicate_filenames(image_paths):
    name_counts = Counter(path.name for path in image_paths)
    duplicate_names = sorted(name for name, count in name_counts.items() if count > 1)

    for name in duplicate_names:
        print(f"Uyari: Ayni dosya adi birden fazla klasorde bulundu: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default=str(TEST_DIR),
        help="Tahmin edilecek gorsellerin klasoru",
    )
    parser.add_argument(
        "--output_file",
        default="Preds.txt",
        help="Cikti Preds.txt dosya yolu",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    image_paths = get_image_paths(input_dir)
    warn_duplicate_filenames(image_paths)

    print(f"Tahmin edilecek gorsel sayisi: {len(image_paths)}")
    model = load_model()
    output_lines = []

    for image_path in image_paths:
        try:
            result = predict_image(image_path, model=model)
            label = result["predicted_label_1_based"]
            output_lines.append(f"{image_path.name} | Pred: {label}")
        except Exception as error:
            print(f"Uyari: {image_path} icin tahmin yapilamadi: {error}")

    output_file.write_text("\n".join(output_lines), encoding="utf-8")
    print(f"Tahmin dosyasi olusturuldu: {output_file}")


if __name__ == "__main__":
    main()
