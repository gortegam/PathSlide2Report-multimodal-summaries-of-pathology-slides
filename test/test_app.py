import pytest
from PIL import Image
from src.app_streamlit import generate_caption

def test_generate_caption_runs():
    # use a blank white image as a dummy input
    img = Image.new("RGB", (224, 224), color="white")
    caption = generate_caption(img)
    assert isinstance(caption, str)
    assert len(caption) > 0
