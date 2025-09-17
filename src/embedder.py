"""
embedder.py
Generate embeddings from images using CLIP.
"""

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class ClipEmbedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def embed_image(self, pil_image: Image.Image):
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            img_emb = self.model.get_image_features(**inputs)
        img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
        return img_emb.cpu().numpy()
