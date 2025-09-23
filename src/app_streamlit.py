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
import csv
from datetime import datetime
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

# --------------------------
# Load metadata
# --------------------------
metadata_csv_path = None
if os.path.exists("data/patches_metadata_enriched.csv"):
    metadata_csv_path = "data/patches_metadata_enriched.csv"
    st.info("Using enriched metadata (diagnosis, sample type, tissue, etc.).")
elif os.path.exists("data/patches_metadata.csv"):
    metadata_csv_path = "data/patches_metadata.csv"
    st.warning("⚠️ Using basic metadata only. Run tcga_metadata_fetcher.py to enrich with diagnosis info.")
else:
    st.warning("⚠️ No TCGA patches found. Falling back to sample data.")

patches_dir = "data/patches"

if metadata_csv_path:
    df_meta = pd.read_csv(metadata_csv_path)
    patch_choice = st.selectbox(
        "Select a patch to analyze:",
        df_meta["patch_file"].tolist()
    )
    patch_path = os.path.join(patches_dir, patch_choice)
    pil_image = Image.open(patch_path).convert("RGB")

    # Use all metadata fields from CSV row
    row = df_meta[df_meta["patch_file"] == patch_choice].iloc[0].to_dict()
    metadata = {k: v for k, v in row.items() if k != "patch_file"}

else:
    # Fallback to sample data
    sample_img = "data/sample_data/sample_slide.png"
    sample_meta = "data/sample_data/metadata.csv"

    if os.path.exists(sample_img) and os.path.exists(sample_meta):
        pil_image = Image.open(sample_img).convert("RGB")
        df = pd.read_csv(sample_meta)
        metadata = df.iloc[0].to_dict()
        st.info("Loaded sample data for demo.")
    else:
        pil_image, metadata = None, {}
        st.error("❌ No data available. Please add either TCGA patches or sample_data.")

# --------------------------
# Main pipeline
# --------------------------
if pil_image:
    st.image(pil_image, caption="Selected patch", use_column_width=True)

    with st.spinner("Generating image caption..."):
        caption = generate_caption(pil_image)
    st.subheader("🖼️ Image Caption")
    st.write(caption)

    with st.spinner("Extracting embeddings..."):
        emb = embed_image(pil_image)
    st.success("Image embedding extracted.")

    if st.button("Generate Report"):
        with st.spinner("Processing patch..."):
            # Add patch to FAISS index
            store.add([emb], [{"metadata": metadata, "caption": caption}])

            # Retrieve similar patches
            retrieved = store.search(emb, k=3)
            retrieved_text = "\n".join(
                [f"- Metadata: {r.get('metadata')}, Caption: {r.get('caption')}" for r in retrieved]
            ) if retrieved else "No similar patches yet."

            # -------- Mode A: Baseline --------
            prompt_A = f"""
You are a pathology assistant.
Patch metadata: {metadata}
BLIP caption: {caption}

Write TWO outputs:
1. A clinical-style summary (2–4 sentences).
2. A layperson summary (1–2 sentences).
"""

            # -------- Mode B: RAG + Metadata --------
            prompt_B = f"""
You are a pathology assistant.
Patch metadata: {metadata}
BLIP caption: {caption}

Here are similar patches retrieved by embedding search:
{retrieved_text}

Write TWO outputs that integrate BOTH the current patch and retrieved metadata:
1. A clinical-style summary (2–4 sentences).
2. A layperson summary (1–2 sentences).
"""

            output_A = query_llm(prompt_A)
            output_B = query_llm(prompt_B)

        # ---------------------------
        # UI Output
        # ---------------------------
        tab1, tab2 = st.tabs(["Baseline (No RAG)", "Improved (RAG + Metadata)"])
        with tab1:
            st.subheader("Without Retrieved Metadata")
            st.write(output_A)
        with tab2:
            st.subheader("With Retrieved Metadata (RAG)")
            st.write(output_B)

        st.markdown(
            "<small>⚠️ This report is AI-generated for demonstration purposes only. Not for clinical use.</small>",
            unsafe_allow_html=True
        )

        # Download both versions
        report_text = f"--- Baseline ---\n{output_A}\n\n--- With RAG + Metadata ---\n{output_B}"
        buf = io.BytesIO(report_text.encode("utf-8"))
        st.download_button(
            label="⬇️ Download Report (Both Versions)",
            data=buf,
            file_name="pathslide2report_comparison.txt",
            mime="text/plain"
        )

        # ---------------------------
        # Save results to log file
        # ---------------------------
        log_file = "logs/run_log.csv"
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["timestamp", "metadata", "caption", "baseline_summary", "rag_summary"])
            writer.writerow([
                datetime.now().isoformat(),
                str(metadata),
                caption,
                output_A,
                output_B
            ])

