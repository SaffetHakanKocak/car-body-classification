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
    "smart fortwo micro car side view",
    "smart fortwo city car side view",
    "smart forfour city car side view",
    "fiat 500 micro car side view",
    "fiat 500 city car side view",
    "toyota iq micro car side view",
    "toyota iq city car side view",
    "citroen c1 city car side view",
    "peugeot 107 city car side view",
    "peugeot 108 city car side view",
    "toyota aygo city car side view",
    "volkswagen up city car side view",
    "seat mii city car side view",
    "skoda citigo city car side view",
    "renault twingo city car side view",
    "hyundai i10 city car side view",
    "kia picanto city car side view",
    "suzuki alto city car side view",
    "suzuki celerio city car side view",
    "chevrolet spark city car side view",
    "mitsubishi mirage city car side view",
    "nissan pixo city car side view",
    "daihatsu cuore city car side view",
    "daihatsu mira city car side view",
    "daihatsu move city car side view",
    "honda n one kei car side view",
    "honda n box kei car side view",
    "suzuki wagon r kei car side view",
    "suzuki alto kei car side view",
    "nissan dayz kei car side view",
    "microcar mgo side view",
    "ligier js50 microcar side view",
    "aixam city microcar side view",
    "aixam coupe microcar side view",
    "citroen ami micro car side view",
    "renault twizy micro car side view",
    "micro electric car side view",
    "small city car side view",
    "tiny city car side view",
]

MAX_RESULTS_PER_QUERY = 200
MIN_WIDTH = 300
MIN_HEIGHT = 200
DOWNLOAD_TIMEOUT = 15
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

EXTERNAL_DATASETS_DIR = BASE_DIR / "external_datasets"
MICRO_CANDIDATES_DIR = EXTERNAL_DATASETS_DIR / "micro_adaylar"
MICRO_SELECTED_DIR = EXTERNAL_DATASETS_DIR / "micro_selected"


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
    MICRO_CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    MICRO_SELECTED_DIR.mkdir(parents=True, exist_ok=True)


def clear_previous_candidates():
    removed_count = 0

    for file_path in MICRO_CANDIDATES_DIR.glob("micro_candidate_*"):
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
        for file_path in MICRO_CANDIDATES_DIR.iterdir()
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
        file_name = f"micro_candidate_{slug}_{saved_index:06d}{extension}"
        target_path = MICRO_CANDIDATES_DIR / file_name

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
    print("MICRO v2 aday gorsel indirme islemi baslatiliyor...")
    prepare_directories()

    ddgs_class = get_ddg_search_class()

    if ddgs_class is None:
        print("Hata: DuckDuckGo image search paketi bulunamadi.")
        print("Kurulum icin: pip install ddgs")
        return

    removed_count = clear_previous_candidates()
    print(f"Onceki micro_candidate_* dosya sayisi temizlendi: {removed_count}")
    print(f"Candidate klasoru: {MICRO_CANDIDATES_DIR}")
    print(f"Selected klasoru: {MICRO_SELECTED_DIR}")

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
    print(f"micro_adaylar toplam aday gorsel sayisi: {count_candidate_images()}")
    print("micro_selected klasoru manuel secim icin hazir.")
    print("Not: Bu script dataset/raw/MICRO klasorune dosya kopyalamaz.")


if __name__ == "__main__":
    main()
