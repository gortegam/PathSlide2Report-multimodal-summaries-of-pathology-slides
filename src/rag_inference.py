"""
rag_inference.py
RAG-style inference: build a prompt with metadata + captions and query an LLM.
"""

import os
import openai

def build_prompt(metadata_list, captions_list):
    md = "\n".join([f"- {m}" for m in metadata_list])
    caps = "\n".join([f"- {c}" for c in captions_list])
    template = f"""
You are a pathology assistant. Use the following slide metadata and image captions
to write a concise clinical summary (2–4 sentences) and a short layperson summary (1–2 sentences).

Slide metadata:
{md}

Image captions:
{caps}

Guidelines:
- Focus on main features.
- Mention stain and tissue.
- If uncertain, say "features suggest..." rather than a definitive diagnosis.
"""
    return template

def query_llm(prompt, model="gpt-4o-mini", max_tokens=250):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        return "⚠️ Please set your OPENAI_API_KEY environment variable."
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a medical summarization assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.4
    )
    return response.choices[0].message.content.strip()
