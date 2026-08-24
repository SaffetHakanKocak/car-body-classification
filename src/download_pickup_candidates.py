import hashlib
import re
import time
from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image

from config import BASE_DIR


SEARCH_QUERIES = [
    "toyota hilux pickup side view",
    "toyota tacoma pickup side view",
    "toyota tundra pickup side view",
    "ford ranger pickup side view",
    "ford f150 pickup side view",
    "ford maverick pickup side view",
    "chevrolet silverado pickup side view",
    "chevrolet colorado pickup side view",
    "gmc sierra pickup side view",
    "gmc canyon pickup side view",
    "dodge ram pickup side view",
    "ram 1500 pickup side view",
    "nissan navara pickup side view",
    "nissan frontier pickup side view",
    "mitsubishi l200 pickup side view",
    "mitsubishi triton pickup side view",
    "isuzu d-max pickup side view",
    "mazda bt-50 pickup side view",
    "volkswagen amarok pickup side view",
    "mercedes x class pickup side view",
    "jeep gladiator pickup side view",
    "honda ridgeline pickup side view",
    "ssangyong musso pickup side view",
    "mahindra pickup side view",
    "great wall poer pickup side view",
    "fiat fullback pickup side view",
    "renault alaskan pickup side view",
    "peugeot landtrek pickup side view",
]

MAX_RESULTS_PER_QUERY = 200
MIN_WIDTH = 300
MIN_HEIGHT = 200
DOWNLOAD_TIMEOUT = 15
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

EXTERNAL_DATASETS_DIR = BASE_DIR / "external_datasets"
PICKUP_CANDIDATES_DIR = EXTERNAL_DATASETS_DIR / "pickup_candidates"
PICKUP_SELECTED_DIR = EXTERNAL_DATASETS_DIR / "pickup_selected"


def get_ddg_search_class():
    try:
        from ddgs import DDGS

        return DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS

            return DDGS
        except ImportError:
            return None


def query_to_slug(query):
    cleaned = query.lower().replace("-", "")
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    return cleaned.strip("_")


def prepare_directories():
    PICKUP_CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    PICKUP_SELECTED_DIR.mkdir(parents=True, exist_ok=True)


def clear_previous_candidates():
    removed_count = 0

    for file_path in PICKUP_CANDIDATES_DIR.glob("pickup_candidate_*"):
        if file_path.is_file():
            file_path.unlink()
            removed_count += 1

    return removed_count


def search_images(ddgs_class, query, max_results):
    try:
        with ddgs_class() as ddgs:
            return list(ddgs.images(query, max_results=max_results))
    except Exception as error:
        print(f"Uyari: '{query}' aramasi basarisiz oldu: {error}")
        return []


def get_image_url(search_result):
    return search_result.get("image") or search_result.get("thumbnail")


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


def extension_from_url(url):
    path = Path(urlparse(url).path)
    extension = path.suffix.lower()

    if extension in SUPPORTED_EXTENSIONS:
        return extension

    return None


def extension_from_image_format(image_format):
    format_to_extension = {
        "JPEG": ".jpg",
        "PNG": ".png",
        "BMP": ".bmp",
        "WEBP": ".webp",
    }

    return format_to_extension.get(image_format)


def validate_image(image_bytes):
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image_format = image.format
            width, height = image.size
            image.verify()

        with Image.open(BytesIO(image_bytes)) as image:
            image.load()

        if width < MIN_WIDTH or height < MIN_HEIGHT:
            return False, None

        extension = extension_from_image_format(image_format)

        if extension not in SUPPORTED_EXTENSIONS:
            return False, None

        return True, extension
    except Exception:
        return False, None


def file_hash(image_bytes):
    return hashlib.sha256(image_bytes).hexdigest()


def count_candidate_images():
    return sum(
        1
        for file_path in PICKUP_CANDIDATES_DIR.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def download_candidates_for_query(ddgs_class, query, known_hashes):
    slug = query_to_slug(query)
    search_results = search_images(ddgs_class, query, MAX_RESULTS_PER_QUERY)

    downloaded_count = 0
    valid_count = 0
    duplicate_count = 0
    saved_index = 1

    print()
    print(f"Sorgu: {query}")
    print(f"Bulunan sonuc sayisi: {len(search_results)}")

    for result in search_results:
        image_url = get_image_url(result)

        if not image_url:
            continue

        try:
            image_bytes = download_bytes(image_url)
            downloaded_count += 1
        except (URLError, TimeoutError, OSError, ValueError) as error:
            print(f"Uyari: Gorsel indirilemedi: {error}")
            continue

        image_hash = file_hash(image_bytes)

        if image_hash in known_hashes:
            duplicate_count += 1
            continue

        is_valid, detected_extension = validate_image(image_bytes)

        if not is_valid:
            continue

        url_extension = extension_from_url(image_url)
        extension = url_extension or detected_extension
        file_name = f"pickup_candidate_{slug}_{saved_index:06d}{extension}"
        target_path = PICKUP_CANDIDATES_DIR / file_name

        try:
            target_path.write_bytes(image_bytes)
        except OSError as error:
            print(f"Uyari: Dosya kaydedilemedi: {error}")
            continue

        known_hashes.add(image_hash)
        valid_count += 1
        saved_index += 1

        time.sleep(0.1)

    print(f"Indirilen gorsel sayisi: {downloaded_count}")
    print(f"Gecerli kabul edilen gorsel sayisi: {valid_count}")
    print(f"Duplicate atlanan gorsel sayisi: {duplicate_count}")

    return {
        "found": len(search_results),
        "downloaded": downloaded_count,
        "valid": valid_count,
        "duplicates": duplicate_count,
    }


def main():
    print("PICK_UP aday gorsel indirme islemi baslatiliyor...")
    prepare_directories()

    ddgs_class = get_ddg_search_class()

    if ddgs_class is None:
        print("Hata: DuckDuckGo image search paketi bulunamadi.")
        print("Kurulum icin: pip install ddgs")
        return

    removed_count = clear_previous_candidates()
    print(f"Onceki pickup_candidate_* dosya sayisi temizlendi: {removed_count}")
    print(f"Candidate klasoru: {PICKUP_CANDIDATES_DIR}")
    print(f"Selected klasoru: {PICKUP_SELECTED_DIR}")

    known_hashes = set()
    total_stats = {
        "found": 0,
        "downloaded": 0,
        "valid": 0,
        "duplicates": 0,
    }

    for query in SEARCH_QUERIES:
        stats = download_candidates_for_query(ddgs_class, query, known_hashes)

        for key in total_stats:
            total_stats[key] += stats[key]

    print()
    print("Genel ozet")
    print(f"Toplam bulunan sonuc sayisi: {total_stats['found']}")
    print(f"Toplam indirilen gorsel sayisi: {total_stats['downloaded']}")
    print(f"Toplam gecerli kabul edilen gorsel sayisi: {total_stats['valid']}")
    print(f"Toplam duplicate atlanan gorsel sayisi: {total_stats['duplicates']}")
    print(f"pickup_candidates toplam aday gorsel sayisi: {count_candidate_images()}")
    print("pickup_selected klasoru manuel secim icin hazir.")
    print("Not: Bu script dataset/raw/PICK_UP klasorune dosya kopyalamaz.")


if __name__ == "__main__":
    main()
