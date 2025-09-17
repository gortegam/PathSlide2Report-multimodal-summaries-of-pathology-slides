"""
captioner.py
Generate captions from pathology slide images using BLIP.
"""

import torch
from transformers import BlipForConditionalGeneration, BlipProcessor
from PIL import Image

class BlipCaptioner:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def caption(self, pil_image: Image.Image, max_length=40):
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_length=max_length)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption
