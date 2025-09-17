"""
data_loader.py
Utility functions to load pathology slide images and associated metadata.
"""

from PIL import Image
import pandas as pd
import os

def load_image(path, target_size=(224,224)):
    """Load and resize a pathology slide image."""
    img = Image.open(path).convert("RGB")
    img = img.resize(target_size)
    return img

def load_metadata(csv_path):
    """Load metadata CSV into a pandas DataFrame."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")
    return pd.read_csv(csv_path)
