# 🧬 PathSlide2Report
**Multimodal Generative AI for Pathology: From Slide + Metadata to Natural-Language Summaries**

---

## 📌 Overview
**PathSlide2Report** is a research/demo project that combines **pathology slide images** and **structured metadata** (e.g., stain type, tissue, magnification) to produce **concise natural-language summaries**.  

The pipeline uses:
- **Vision encoders (CLIP/BLIP)** → extract image features or generate captions.
- **Metadata ingestion** → contextual slide information.
- **Retrieval-Augmented Generation (RAG)** → combine image embeddings + metadata for context.
- **LLM (e.g., GPT, LLaMA, Mistral)** → generate both **clinical-style summaries** and **layperson explanations**.

⚠️ **Disclaimer**: This project is for **research and educational purposes only**. It is **not intended for clinical use or medical decision-making**.  

---

## 🚀 Features
- 🔍 Image embedding with **CLIP** or **BLIP**
- 📝 Automatic caption generation from pathology slides
- 📑 Integration of slide metadata (CSV or JSON)
- 📦 Embedding storage & retrieval using **FAISS** or **ChromaDB**
- 🤖 **RAG-powered summaries** (clinical + lay)
- 🌐 **Streamlit demo app** for easy testing
- 📊 Evaluation with BLEU/ROUGE and clinician-style ratings
- ✅ Ethical safeguards (hallucination checks, disclaimers)

---

## 🛠️ Tech Stack
- [Python 3.10+](https://www.python.org/)  
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) (CLIP, BLIP)  
- [PyTorch](https://pytorch.org/)  
- [FAISS](https://github.com/facebookresearch/faiss) or [ChromaDB](https://www.trychroma.com/)  
- [LangChain](https://www.langchain.com/) (optional for RAG pipelines)  
- [OpenAI API](https://platform.openai.com/) or local LLMs (e.g., LLaMA/Mistral)  
- [Streamlit](https://streamlit.io/) for the demo app  

---

## 📂 Repository Structure

pathslide2report/
├─ data/ # sample or synthetic slides + metadata
├─ notebooks/
│ └─ exploratory.ipynb # walkthrough of data ingestion → summary
├─ src/
│ ├─ data_loader.py # load images + metadata
│ ├─ embedder.py # CLIP embeddings
│ ├─ captioner.py # BLIP captioning
│ ├─ vectorstore.py # FAISS/ChromaDB retriever
│ ├─ rag_inference.py # summary generation pipeline
│ └─ app_streamlit.py # demo app
├─ tests/ # simple unit tests
├─ requirements.txt
├─ README.md
└─ demo_gifs/ # screenshots or screen recordings



---

## 📊 Sample Data
Since real pathology data often contains PHI, this repo includes **synthetic or public-domain samples** (e.g., from [PatchCamelyon](https://github.com/basveeling/pcam) or simulated H&E slides).  

**Metadata format example (`metadata.csv`):**
```csv
slide_id,tissue,stain,magnification,accession_date
slide_0001,Liver,H&E,40x,2025-01-01
slide_0002,Lung,H&E,20x,2025-01-05

