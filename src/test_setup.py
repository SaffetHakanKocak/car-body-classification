import torch
import torchvision
import numpy as np
import pandas as pd
import sklearn
import cv2
import streamlit as st

from config import CLASS_NAMES, DEVICE


def main():
    print("Python ortami duzgun calisiyor.")
    print("Torch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", DEVICE)

    print("NumPy version:", np.__version__)
    print("Pandas version:", pd.__version__)
    print("Scikit-learn version:", sklearn.__version__)
    print("OpenCV version:", cv2.__version__)
    print("Streamlit version:", st.__version__)

    print(f"Sinif sayisi: {len(CLASS_NAMES)}")
    print("Sinif isimleri:", CLASS_NAMES)


if __name__ == "__main__":
    main()
