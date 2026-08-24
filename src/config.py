from pathlib import Path

import torch


# Proje ana dizini
BASE_DIR = Path(__file__).resolve().parent.parent

# Veri klasorleri
DATASET_DIR = BASE_DIR / "dataset"
RAW_DATA_DIR = DATASET_DIR / "raw"
PROCESSED_DATA_DIR = DATASET_DIR / "processed"
TRAIN_DIR = PROCESSED_DATA_DIR / "train"
VAL_DIR = PROCESSED_DATA_DIR / "val"
TEST_DIR = PROCESSED_DATA_DIR / "test"

# Cikti ve model klasorleri
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Sinif isimleri bu sirayla kullanilacak
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

NUM_CLASSES = len(CLASS_NAMES)

# Egitim ayarlari
MODEL_NAME = "efficientnet_b2"
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.4

# Dosya yollari
BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

# CUDA varsa ekran karti, yoksa CPU kullanilir
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
