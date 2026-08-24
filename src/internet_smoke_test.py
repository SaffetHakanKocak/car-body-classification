import argparse
import csv
import hashlib
import re
import sys
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image

from config import BASE_DIR, CLASS_NAMES
from predict import load_model, predict_image


DEFAULT_SAMPLES_PER_CLASS = 12
DEFAULT_MAX_RESULTS_PER_QUERY = 80
MIN_WIDTH = 300
MIN_HEIGHT = 200
DOWNLOAD_TIMEOUT = 15
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TEST_DIR = BASE_DIR / "external_datasets" / "internet_smoke_test"
OUTPUT_CSV = BASE_DIR / "outputs" / "internet_smoke_test_results.csv"

SEARCH_QUERIES = {
    "SUV": [
        "toyota rav4 suv side view real photo",
        "honda cr-v suv front three quarter view real photo",
        "nissan qashqai suv rear three quarter view real photo",
        "hyundai tucson suv front view real photo",
    ],
    "VAN": [
        "ford transit van side view real photo",
        "mercedes sprinter van front three quarter view real photo",
        "volkswagen transporter van rear three quarter view real photo",
        "renault trafic van front view real photo",
    ],
    "STATION_WAGON": [
        "volvo v60 wagon side view real photo",
        "audi a4 avant front three quarter view real photo",
        "bmw 3 series touring rear three quarter view real photo",
        "skoda octavia combi front view real photo",
    ],
    "MICRO": [
        "smart fortwo micro car side view real photo",
        "toyota iq micro car front three quarter view real photo",
        "citroen ami micro car rear three quarter view real photo",
        "renault twizy micro car front view real photo",
    ],
    "OPEN_WHEEL": [
        "formula 1 car side view real photo",
        "indycar race car front three quarter view real photo",
        "formula 2 car rear three quarter view real photo",
        "single seater race car front view real photo",
    ],
    "SEDAN": [
        "toyota corolla sedan side view real photo",
        "bmw 3 series sedan front three quarter view real photo",
        "honda civic sedan rear three quarter view real photo",
        "mercedes c class sedan front view real photo",
    ],
    "HATCHBACK": [
        "volkswagen golf hatchback side view real photo",
        "ford fiesta hatchback front three quarter view real photo",
        "renault clio hatchback rear three quarter view real photo",
        "peugeot 208 hatchback front view real photo",
    ],
    "PICK_UP": [
        "toyota hilux pickup side view real photo",
        "ford ranger pickup front three quarter view real photo",
        "mitsubishi l200 pickup rear three quarter view real photo",
        "nissan navara pickup front view real photo",
    ],
}


def get_ddgs_class():
    try:
        from ddgs import DDGS

        return DDGS
    except ImportError as error:
        raise ImportError("ddgs paketi gerekli: pip install ddgs") from error


def query_to_slug(query):
    cleaned = query.lower().replace("-", " ")
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    return cleaned.strip("_")


def clear_old_test_images():
    if not TEST_DIR.exists():
        return

    for file_path in TEST_DIR.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            file_path.unlink()


def search_images(ddgs_class, query, max_results_per_query):
    try:
        with ddgs_class() as ddgs:
            return list(ddgs.images(query, max_results=max_results_per_query))
    except Exception as error:
        print(f"Uyari: arama basarisiz: {query} - {error}")
        return []


def get_image_url(result):
    return result.get("image") or result.get("thumbnail")


def download_bytes(url):
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        },
    )
    with urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
        return response.read()


def extension_from_format(image_format):
    mapping = {
        "JPEG": ".jpg",
        "PNG": ".png",
        "BMP": ".bmp",
        "WEBP": ".webp",
    }
    return mapping.get(image_format)


def validate_image(image_bytes):
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image_format = image.format
            width, height = image.size
            image.verify()

        if width < MIN_WIDTH or height < MIN_HEIGHT:
            return False, None

        extension = extension_from_format(image_format)
        if extension not in SUPPORTED_EXTENSIONS:
            return False, None

        return True, extension
    except Exception:
        return False, None


def file_hash(image_bytes):
    return hashlib.sha256(image_bytes).hexdigest()


def download_samples(samples_per_class, max_results_per_query):
    ddgs_class = get_ddgs_class()
    known_hashes = set()
    downloaded = {}

    clear_old_test_images()
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    for class_index, class_name in enumerate(CLASS_NAMES, start=1):
        class_dir = TEST_DIR / f"{class_index}_{class_name}"
        class_dir.mkdir(parents=True, exist_ok=True)
        saved_count = 0

        print()
        print(f"Sinif: {class_name}")

        queries = SEARCH_QUERIES[class_name]

        for query_index, query in enumerate(queries):
            if saved_count >= samples_per_class:
                break

            results = search_images(ddgs_class, query, max_results_per_query)
            print(f"  Sorgu: {query} | sonuc: {len(results)}")

            remaining_queries = len(queries) - query_index
            query_target = max(
                1,
                (samples_per_class - saved_count + remaining_queries - 1)
                // remaining_queries,
            )
            saved_for_query = 0

            for result in results:
                if saved_count >= samples_per_class or saved_for_query >= query_target:
                    break

                url = get_image_url(result)
                if not url:
                    continue

                try:
                    image_bytes = download_bytes(url)
                except Exception:
                    continue

                image_hash = file_hash(image_bytes)
                if image_hash in known_hashes:
                    continue

                is_valid, extension = validate_image(image_bytes)
                if not is_valid:
                    continue

                known_hashes.add(image_hash)
                saved_count += 1
                saved_for_query += 1
                file_name = f"{class_index}_{class_name.lower()}_{saved_count:02d}{extension}"
                (class_dir / file_name).write_bytes(image_bytes)

        downloaded[class_name] = saved_count
        print(f"  Kaydedilen gorsel: {saved_count}")

    return downloaded


def run_predictions():
    model = load_model()
    rows = []

    image_paths = sorted(
        [
            path
            for path in TEST_DIR.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ],
        key=lambda path: (path.parent.name, path.name),
    )

    for image_path in image_paths:
        true_class = image_path.parent.name.split("_", 1)[1]
        result = predict_image(image_path, model=model)
        predicted_class = result["predicted_class"]

        rows.append(
            {
                "file_name": image_path.name,
                "image_path": str(image_path),
                "true_class": true_class,
                "predicted_class": predicted_class,
                "predicted_label_1_based": result["predicted_label_1_based"],
                "confidence": result["confidence"],
                "is_correct": true_class == predicted_class,
            }
        )

    return rows


def save_results(rows):
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    total = len(rows)
    correct = sum(1 for row in rows if row["is_correct"])
    accuracy = correct / total if total else 0.0

    print()
    print("Internet smoke test sonucu")
    print(f"Toplam gorsel: {total}")
    print(f"Dogru tahmin: {correct}")
    print(f"Yanlis tahmin: {total - correct}")
    print(f"Accuracy: {accuracy:.4f}")

    print()
    print("Sinif bazinda:")
    for class_name in CLASS_NAMES:
        class_rows = [row for row in rows if row["true_class"] == class_name]
        class_total = len(class_rows)
        class_correct = sum(1 for row in class_rows if row["is_correct"])
        class_accuracy = class_correct / class_total if class_total else 0.0
        print(f"{class_name}: {class_correct}/{class_total} ({class_accuracy:.2%})")

    wrong_rows = [row for row in rows if not row["is_correct"]]
    if wrong_rows:
        print()
        print("Yanlis tahminler:")
        for row in wrong_rows:
            print(
                f"{row['file_name']}: {row['true_class']} -> "
                f"{row['predicted_class']} ({row['confidence']:.2%})"
            )

    print()
    print(f"CSV kaydedildi: {OUTPUT_CSV}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=DEFAULT_SAMPLES_PER_CLASS,
        help="Her sinif icin indirilecek gorsel sayisi",
    )
    parser.add_argument(
        "--max_results_per_query",
        type=int,
        default=DEFAULT_MAX_RESULTS_PER_QUERY,
        help="Her sorgu icin maksimum arama sonucu",
    )
    args = parser.parse_args()

    try:
        downloaded = download_samples(args.samples_per_class, args.max_results_per_query)
    except ImportError as error:
        print(error)
        sys.exit(1)

    if any(count < args.samples_per_class for count in downloaded.values()):
        print()
        print("Uyari: Bazi siniflarda hedef gorsel sayisina ulasilamadi.")

    rows = run_predictions()

    if not rows:
        print("Uyari: Test edilecek gorsel bulunamadi.")
        return

    save_results(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()
