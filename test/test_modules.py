"""
Basic unit tests for PathSlide2Report modules.
These tests verify that imports work and core functions run without errors.
"""

import pytest
from PIL import Image
import numpy as np

# Import your modules
from src.data_loader import load_image, load_metadata
from src.embedder import ClipEmbedder
from src.captioner import BlipCaptioner
from src.vectorstore import FaissStore
from src.rag_inference import build_prompt


def test_load_image(tmp_path):
    # Create a dummy image
    img_path = tmp_path / "dummy.png"
    Image.new("RGB", (64, 64), color="white").save(img_path)

    img = load_image(str(img_path))
    assert isinstance(img, Image.Image)


def test_load_metadata(tmp_path):
    # Create a dummy CSV
    csv_path = tmp_path / "meta.csv"
    csv_path.write_text("slide_id,tissue\ns1,Liver\n")
    df = load_metadata(str(csv_path))
    assert "tissue" in df.columns


def test_clip_embedder_runs():
    embedder = ClipEmbedder()
    img = Image.new("RGB", (224, 224), color="white")
    emb = embedder.embed_image(img)
    assert isinstance(emb, np.ndarray)
    assert emb.shape[1] > 0


def test_blip_captioner_runs():
    captioner = BlipCaptioner()
    img = Image.new("RGB", (224, 224), color="white")
    caption = captioner.caption(img)
    assert isinstance(caption, str)
    assert len(caption) > 0


def test_faiss_store_add_and_search():
    store = FaissStore(dim=4)
    emb = np.random.rand(1, 4).astype("float32")
    store.add([emb], [{"id": "slide1"}])
    results = store.search(emb, k=1)
    assert results[0]["id"] == "slide1"


def test_build_prompt():
    metadata_list = ["slide1: tissue=Liver, stain=H&E"]
    captions_list = ["a sample caption"]
    prompt = build_prompt(metadata_list, captions_list)
    assert "Liver" in prompt
    assert "sample caption" in prompt


def test_app_imports():
    """
    Ensure the Streamlit app file can be imported without errors.
    """
    import importlib
    module = importlib.import_module("src.app_streamlit")
    assert module is not None
