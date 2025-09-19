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
import io
from src.vectorstore import FaissStore

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

# Initialize FAISS vector store
store = FaissStore(dim=512)

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

def query_llm(prompt, model="gpt-4o-mini", max_tokens=350):
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
    value=""
)

# Fallback: use sample_data if nothing uploaded
if not uploaded_file:
    st.info("No file uploaded — using sample data from `sample_data/`.")
    try:
        pil_image = Image.open("sample_data/sample_slide.png").convert("RGB")
        df = pd.read_csv("sample_data/metadata.csv")
        metadata_input = "\n".join([f"{col}: {df.iloc[0][col]}" for col in df.columns if col != "slide_id"])
    except Exception as e:
        st.error(f"Sample data not found: {e}")
        pil_image = None
else:
    pil_image = Image.open(uploaded_file).convert("RGB")

if pil_image:
    st.image(pil_image, caption="Slide image", use_column_width=True)

    with st.spinner("Generating image caption..."):
        caption = generate_caption(pil_image)
    st.subheader("🖼️ Image Caption")
    st.write(caption)

    with st.spinner("Extracting embeddings..."):
        emb = embed_image(pil_image)
    st.success("Image embedding extracted.")

    # Parse metadata
    metadata = {}
    for line in metadata_input.split("\n"):
        if ":" in line:
            k, v = line.split(":", 1)
            metadata[k.strip()] = v.strip()

    if st.button("Generate Report"):
        with st.spinner("Processing slide..."):
            # Add current slide to FAISS index
            store.add([emb], [metadata])

            # Retrieve similar slides
            retrieved = store.search(emb, k=3)
            retrieved_text = "\n".join(
                [f"- {r}" for r in retrieved]
            ) if retrieved else "No similar slides yet."

            # Build RAG prompt
            prompt = f"""
You are a pathology assistant.

The current slide has metadata: {metadata}
Image caption: {caption}

Here are the most similar past slides:
{retrieved_text}

Write TWO outputs:
1. A clinical-style summary (2–4 sentences).
2. A layperson summary (1–2 sentences).
"""

            output = query_llm(prompt)

        # Split summaries
        clinical_summary = "Not found"
        lay_summary = "Not found"
        if "Clinical" in output or "clinical" in output.lower():
            parts = output.split("2.")
            if len(parts) == 2:
                clinical_summary = parts[0].replace("1.", "").strip()
                lay_summary = parts[1].strip()
        else:
            clinical_summary = output

        # Tabs
        tab1, tab2 = st.tabs(["Clinical Summary", "Lay Summary"])
        with tab1:
            st.write(clinical_summary)
        with tab2:
            st.write(lay_summary)

        # Disclaimer
        st.markdown(
            "<small>⚠️ This report is AI-generated for demonstration purposes only. Not for clinical use.</small>",
            unsafe_allow_html=True
        )

        # Download button
        report_text = f"Clinical Summary:\n{clinical_summary}\n\nLay Summary:\n{lay_summary}"
        buf = io.BytesIO(report_text.encode("utf-8"))
        st.download_button(
            label="⬇️ Download Report",
            data=buf,
            file_name="pathslide2report_summary.txt",
            mime="text/plain"
        )
