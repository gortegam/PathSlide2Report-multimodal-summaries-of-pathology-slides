# src/app_streamlit.py

import streamlit as st
from PIL import Image
import pandas as pd
import torch
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    CLIPProcessor, CLIPModel
)
import openai
import os

# --------------------------
# Setup models
# --------------------------
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

@st.cache_resource
def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return processor, model

blip_processor, blip_model = load_blip()
clip_processor, clip_model = load_clip()

device = "cuda" if torch.cuda.is_available() else "cpu"
blip_model = blip_model.to(device)
clip_model = clip_model.to(device)

# --------------------------
# Helper functions
# --------------------------
def generate_caption(pil_image):
    inputs = blip_processor(images=pil_image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs, max_length=40)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

def embed_image(pil_image):
    inputs = clip_processor(images=pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        img_emb = clip_model.get_image_features(**inputs)
    return img_emb.cpu().numpy()

def build_prompt(metadata, caption):
    meta_str = "\n".join([f"- {k}: {v}" for k, v in metadata.items() if v])
    prompt = f"""
You are a pathology assistant. 
Write a short clinical-style summary (2–4 sentences) and a layperson summary (1–2 sentences) 
based on the metadata and image caption below. 
Be cautious—if uncertain, say "features suggest..." instead of making a definitive diagnosis.

Metadata:
{meta_str}

Image caption:
{caption}

Summaries:
"""
    return prompt

def query_llm(prompt, model="gpt-4o-mini", max_tokens=250):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        return "⚠️ Please set OPENAI_API_KEY environment variable."
    response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a medical summarization assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.4
    )
    return response.choices[0].message.content.strip()

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="PathSlide2Report", layout="wide")
st.title("🧬 PathSlide2Report — Multimodal Gen AI Demo")

uploaded_file = st.file_uploader("Upload a pathology slide (PNG/JPG/TIF)", type=["png", "jpg", "jpeg", "tif"])
metadata_input = st.text_area(
    "Enter metadata (key:value pairs, one per line)",
    value="tissue: liver\nstain: H&E\nmagnification: 40x\naccession_date: 2025-01-01"
)

if uploaded_file:
    pil_image = Image.open(uploaded_file).convert("RGB")
    st.image(pil_image, caption="Uploaded slide", use_column_width=True)

    with st.spinner("Generating image caption..."):
        caption = generate_caption(pil_image)
    st.subheader("🖼️ Image Caption")
    st.write(caption)

    with st.spinner("Extracting embeddings..."):
        _ = embed_image(pil_image)
    st.success("Image embedding extracted.")

    # Parse metadata
    metadata = {}
    for line in metadata_input.split("\n"):
        if ":" in line:
            k, v = line.split(":", 1)
            metadata[k.strip()] = v.strip()

    # Build prompt + query LLM
    if st.button("Generate Report"):
        with st.spinner("Querying LLM..."):
            prompt = build_prompt(metadata, caption)
            summary = query_llm(prompt)
        st.subheader("📝 Generated Report")
        st.write(summary)
